"""E7 / C: cycle-based gate-level simulator for the mapped (Nangate45) netlists, bit-parallel over time.

A net is a numpy uint64 vector whose bit t is its settled value in clock cycle t (all cycles of a stream
are evaluated at once).  Cells are evaluated level by level, grouped by cell type (liberty functions).
Flip-flops: Q(t) = D(t-1); feedback through flops (output-stationary accumulator) is resolved by
fixed-point iteration from an initial guess (exact after convergence: Q(t) == D(t-1) for all t).

Toggle counts:
  zero-delay : transitions of the settled value between consecutive cycles (no glitches);
  unit-delay : every cell has delay 1, all cells update synchronously from the previous sub-step
               (transport delay), sub-steps until quiescence; counts glitches (an upper-side
               estimate: inertial filtering of short pulses is ignored).
Segment starts (artificial joins of independent streams) are excluded from the counts.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from liberty import read_liberty, func_to_py


def pack_bits(bits: np.ndarray, nwords: int) -> np.ndarray:
    """bool/0-1 array (n,) -> uint64 (nwords,), bit t of the stream = bit (t%64) of word t//64."""
    b = np.zeros(nwords * 64, dtype=np.uint8)
    b[: bits.size] = bits
    return np.packbits(b, bitorder="little").view(np.uint64)


def unpack_bits(words: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(words.view(np.uint8), bitorder="little")[:n]


def shift1(x: np.ndarray) -> np.ndarray:
    """y[t] = x[t-1] along the packed time axis (last axis), y[0] = 0."""
    y = x << np.uint64(1)
    y[..., 1:] |= x[..., :-1] >> np.uint64(63)
    return y


class GateNetlist:
    def __init__(self, net_json: Path, liberty: Path, top: str | None = None):
        j = json.loads(Path(net_json).read_text())
        mods = j["modules"]
        if top is None:
            top = [k for k, m in mods.items() if m.get("attributes", {}).get("top")][0]
        self.top = top
        m = mods[top]
        lib = read_liberty(str(liberty))["cells"]
        self.lib = lib
        idx = {"0": 0, "1": 1}
        def nid(b):
            k = str(b)
            if k not in idx:
                idx[k] = len(idx)
            return idx[k]
        self.ports = {}
        for name, p in m["ports"].items():
            self.ports[name] = (p["direction"], [nid(b) for b in p["bits"]])
        self.flops = []        # dict(name, D, Q, QN)
        comb = []              # (cell name, type, {pin: net})
        for cname, c in m["cells"].items():
            t = c["type"]
            lc = lib[t]
            conns = {pin: nid(bits[0]) for pin, bits in c["connections"].items() if bits}
            if lc["ff"] is not None:
                ff = lc["ff"]
                assert ff["next_state"] == "D" and lc["pins"].get("CK"), f"unsupported flop {t}"
                self.flops.append(dict(name=cname, type=t, D=conns["D"], Q=conns.get("Q"), QN=conns.get("QN")))
            else:
                comb.append((cname, t, conns))
        self.n = len(idx)
        # flop Q -> output port bit (directly or through one buffer inserted by insbuf)
        port_of = {}
        for name, (d, bits) in self.ports.items():
            if d == "output":
                for i, b in enumerate(bits):
                    port_of[b] = (name, i)
        buf_out = {}
        for cname, t, conns in comb:
            if t.startswith("BUF_") and "A" in conns and "Z" in conns:
                buf_out.setdefault(conns["A"], []).append(conns["Z"])
        for f in self.flops:
            q = f["Q"]
            f["port"] = port_of.get(q) or next((port_of[z] for z in buf_out.get(q, []) if z in port_of), None)
        self.names = {v: k for k, v in idx.items()}
        for nm, nn in m.get("netnames", {}).items():
            for i, b in enumerate(nn["bits"]):
                if str(b) in idx and not nm.startswith("$"):
                    self.names[idx[str(b)]] = f"{nm}[{i}]" if len(nn["bits"]) > 1 else nm
        # one entry per (cell, output pin)
        outs = []
        for cname, t, conns in comb:
            for pin, pd in lib[t]["pins"].items():
                if pd["direction"] == "output" and pin in conns and pd["function"]:
                    ins = sorted(p for p, q in lib[t]["pins"].items() if q["direction"] == "input")
                    outs.append((cname, t, pin, conns[pin], [conns.get(p, 0) for p in ins], ins, pd["function"]))
        self.outs = outs
        self.driver = {o[3]: k for k, o in enumerate(outs)}
        self.sources = {0, 1}
        for name, (d, bits) in self.ports.items():
            if d == "input":
                self.sources |= set(bits)
        for f in self.flops:
            self.sources |= {f["Q"], f["QN"]} - {None}
        # levelise
        lev = {s: 0 for s in self.sources}
        order = []
        pending = list(range(len(outs)))
        while pending:
            nxt = []
            for k in pending:
                ins = outs[k][4]
                if all(i in lev for i in ins):
                    lev[outs[k][3]] = 1 + max([lev[i] for i in ins] or [0])
                    order.append(k)
                else:
                    nxt.append(k)
            if len(nxt) == len(pending):
                raise RuntimeError("combinational loop or undriven net")
            pending = nxt
        self.level = lev
        self.maxlev = max(lev.values())
        # group by (level, type, pin) for zero-delay, by (type, pin) for unit-delay
        self.groups_lv = defaultdict(list)
        self.groups_t = defaultdict(list)
        for k, o in enumerate(outs):
            self.groups_lv[(lev[o[3]], o[1], o[2])].append(k)
            self.groups_t[(o[1], o[2])].append(k)
        self.funcs = {}
        for (t, pin) in self.groups_t:
            o = outs[self.groups_t[(t, pin)][0]]
            expr = func_to_py(o[6])
            args = ", ".join(o[5])
            self.funcs[(t, pin)] = eval(f"lambda {args}: {expr}", {"ONE": np.uint64(~np.uint64(0)), "ZERO": np.uint64(0)})
        self._prep()

    def _prep(self):
        self.lv_items = []
        for (lv, t, pin), ks in sorted(self.groups_lv.items()):
            outs = np.array([self.outs[k][3] for k in ks])
            ins = np.array([self.outs[k][4] for k in ks]).reshape(len(ks), -1)
            self.lv_items.append((self.funcs[(t, pin)], outs, ins))
        self.t_items = []
        for (t, pin), ks in self.groups_t.items():
            outs = np.array([self.outs[k][3] for k in ks])
            ins = np.array([self.outs[k][4] for k in ks]).reshape(len(ks), -1)
            self.t_items.append((self.funcs[(t, pin)], outs, ins))
        self.comb_nets = np.array([o[3] for o in self.outs])

    # ------------------------------------------------------------------------------------------
    def new_state(self, nwords: int) -> np.ndarray:
        V = np.zeros((self.n, nwords), dtype=np.uint64)
        V[1] = ~np.uint64(0)
        return V

    def set_port(self, V, name: str, values: np.ndarray, nbits: int):
        """values: int64 array (stream length) -> port bits."""
        _, bits = self.ports[name]
        nw = V.shape[1]
        for i, b in enumerate(bits):
            if b > 1:
                V[b] = pack_bits(((values >> i) & 1).astype(np.uint8), nw)

    def get_nets(self, V, nets, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(nets):
            out |= unpack_bits(V[b], n).astype(np.int64) << i
        return out

    def flop_nets(self, port: str):
        """D and Q nets of the flops behind output port `port`, ordered by bit."""
        fl = sorted((f for f in self.flops if f["port"] and f["port"][0] == port), key=lambda f: f["port"][1])
        return [f["D"] for f in fl], [f["Q"] for f in fl], [f for f in fl]

    def simulate(self, port_streams: dict, q_init: dict, nbits: int, seg_mask, maxit: int = 30):
        """Sequential simulation of a stream (bit-parallel over time).  port_streams: input port ->
        int64 values per cycle; q_init: output port of a register -> initial guess of Q per cycle.
        Iterates Q(t) = D(t-1) to the fixed point.  Returns (V, iterations)."""
        nw = -(-nbits // 64)
        V = self.new_state(nw)
        for name, vals in port_streams.items():
            self.set_port(V, name, vals, nbits)
        D = np.array([f["D"] for f in self.flops], dtype=np.int64)
        Q = np.array([f["Q"] for f in self.flops], dtype=np.int64)
        QN = [(k, f["QN"]) for k, f in enumerate(self.flops) if f["QN"] is not None]
        for f in self.flops:
            if f["port"] and f["port"][0] in q_init:
                V[f["Q"]] = pack_bits(((q_init[f["port"][0]] >> f["port"][1]) & 1).astype(np.uint8), nw)
        full = pack_bits(np.ones(nbits, np.uint8), nw)
        for it in range(maxit):
            for k, qn in QN:
                V[qn] = ~V[Q[k]]
            self.settle(V)
            if not len(D):
                return V, it + 1
            Qn = (shift1(V[D]) & ~seg_mask) | (V[D] & seg_mask)
            if np.array_equal(Qn & full, V[Q] & full):
                return V, it + 1
            V[Q] = Qn
        raise RuntimeError("flip-flop fixed point not reached")

    def settle(self, V):
        for f, outs, ins in self.lv_items:
            V[outs] = f(*[V[ins[:, i]] for i in range(ins.shape[1])])

    def unit_delay_toggles(self, V_settled, seg_start_mask, valid_mask):
        """Glitch-inclusive toggle counts of all combinational outputs (unit-delay, transport)."""
        prev = (shift1(V_settled) & ~seg_start_mask) | (V_settled & seg_start_mask)
        cur = prev.copy()
        src = np.array(sorted(self.sources))
        cur[src] = V_settled[src]
        tog = np.zeros(self.n, dtype=np.int64)
        for step in range(self.maxlev + 2):
            new = cur.copy()
            for f, outs, ins in self.t_items:
                new[outs] = f(*[cur[ins[:, i]] for i in range(ins.shape[1])])
            d = (new[self.comb_nets] ^ cur[self.comb_nets]) & valid_mask
            if not d.any():
                break
            tog[self.comb_nets] += np.bitwise_count(d).sum(axis=1).astype(np.int64)
            cur = new
        assert np.array_equal(cur[self.comb_nets] & valid_mask, V_settled[self.comb_nets] & valid_mask), \
            "unit-delay simulation did not settle to the zero-delay values"
        return tog

    def zero_delay_toggles(self, V, valid_mask):
        prev = shift1(V)
        return np.bitwise_count((V ^ prev) & valid_mask).sum(axis=1).astype(np.int64)

    def ones(self, V, full_mask):
        return np.bitwise_count(V & full_mask).sum(axis=1).astype(np.int64)


class Activity:
    """Accumulates per-net toggle counts / high time over many evaluations of one netlist."""

    def __init__(self, g: GateNetlist):
        self.g = g
        self.tog0 = np.zeros(g.n, dtype=np.int64)
        self.togu = np.zeros(g.n, dtype=np.int64)
        self.ones = np.zeros(g.n, dtype=np.int64)
        self.cycles = 0          # PE-cycles counted for zero-delay toggles
        self.cycles_u = 0        # PE-cycles counted for unit-delay toggles

    def add(self, V, valid_mask, full_mask, ncycles, unit_delay: bool, seg_mask=None):
        self.tog0 += self.g.zero_delay_toggles(V, valid_mask)
        self.ones += self.g.ones(V, full_mask)
        self.cycles += ncycles
        if unit_delay:
            tu = self.g.unit_delay_toggles(V, seg_mask, valid_mask)
            # sources (inputs, flop outputs) have no glitches: use their zero-delay toggles
            z = self.g.zero_delay_toggles(V, valid_mask)
            src = np.array(sorted(self.g.sources))
            tu[src] = z[src]
            self.togu += tu
            self.cycles_u += ncycles

    def merge(self, other: "Activity"):
        self.tog0 += other.tog0; self.togu += other.togu; self.ones += other.ones
        self.cycles += other.cycles; self.cycles_u += other.cycles_u

    def to_dict(self):
        g = self.g
        pins = {}
        # driver pin of every net: comb cell output pin or flop Q/QN or input port
        for k, o in enumerate(g.outs):
            pins[o[3]] = f"{o[0]}/{o[2]}"
        for f in g.flops:
            if f["Q"] is not None:
                pins[f["Q"]] = f"{f['name']}/Q"
            if f["QN"] is not None:
                pins[f["QN"]] = f"{f['name']}/QN"
        ports = {}
        for name, (d, bits) in g.ports.items():
            if d == "input":
                for i, b in enumerate(bits):
                    ports[b] = name if len(bits) == 1 else f"{name}[{i}]"
        nets = []
        for n in range(2, g.n):
            if n in pins or n in ports:
                nets.append(dict(net=n, pin=pins.get(n), port=ports.get(n), tog0=int(self.tog0[n]),
                                 togu=int(self.togu[n]), ones=int(self.ones[n])))
        return dict(cycles=self.cycles, cycles_unit_delay=self.cycles_u, nets=nets)

"""Minimal Liberty reader: cell area, pin direction/function/capacitance, flip-flop groups, leakage.

Enough for the zero-/unit-delay gate-level simulator (gatesim.py); timing/power tables are left to
OpenSTA (sta_power.py).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_TOK = re.compile(r'"(?:[^"\\]|\\[\s\S])*"|[{}();:,]|\\\n|[^\s{}();:,"]+')


def _parse(text: str):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    toks = [t for t in _TOK.findall(text) if t not in ("\\\n", "\\")]
    pos = 0

    def group():
        nonlocal pos
        node = {"attrs": {}, "groups": []}
        while pos < len(toks):
            t = toks[pos]
            if t == "}":
                pos += 1
                return node
            name = t
            pos += 1
            if pos < len(toks) and toks[pos] == ":":
                pos += 1
                val = []
                while toks[pos] != ";":
                    val.append(toks[pos])
                    pos += 1
                pos += 1
                node["attrs"][name] = " ".join(val).strip('"')
            elif pos < len(toks) and toks[pos] == "(":
                pos += 1
                args = []
                while toks[pos] != ")":
                    if toks[pos] != ",":
                        args.append(toks[pos].strip('"'))
                    pos += 1
                pos += 1
                if pos < len(toks) and toks[pos] == "{":
                    pos += 1
                    sub = group()
                    sub["type"], sub["args"] = name, args
                    node["groups"].append(sub)
                elif pos < len(toks) and toks[pos] == ";":
                    pos += 1
                    node["attrs"].setdefault(name, args)
            else:
                pass
        return node

    return group()


@lru_cache(maxsize=4)
def read_liberty(path: str) -> dict:
    root = _parse(Path(path).read_text())
    lib = root["groups"][0]
    cells = {}
    for g in lib["groups"]:
        if g["type"] != "cell":
            continue
        c = dict(name=g["args"][0], area=float(g["attrs"].get("area", 0)), pins={}, ff=None,
                 leakage_nW=float(g["attrs"].get("cell_leakage_power", 0)))
        for p in g["groups"]:
            if p["type"] == "pin":
                a = p["attrs"]
                c["pins"][p["args"][0]] = dict(direction=a.get("direction"), function=a.get("function"),
                                               capacitance=float(a.get("capacitance", 0) or 0))
            elif p["type"] in ("ff", "latch"):
                a = p["attrs"]
                c["ff"] = dict(kind=p["type"], q=p["args"][0], qn=p["args"][1] if len(p["args"]) > 1 else None,
                               next_state=a.get("next_state"), clocked_on=a.get("clocked_on"))
        cells[c["name"]] = c
    return dict(name=lib["args"][0], cells=cells, attrs=lib["attrs"])


def func_to_py(fn: str) -> str:
    """Liberty boolean function -> python expression on (numpy uint64) variables named like the pins."""
    s = fn.replace("!", "~").replace("+", "|").replace("*", "&")
    s = re.sub(r"(\w+)'", r"(~\1)", s)
    s = re.sub(r"\b1\b", "ONE", s)
    s = re.sub(r"\b0\b", "ZERO", s)
    return s

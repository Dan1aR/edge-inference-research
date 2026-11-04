// E7: systolic arrays of BF16 multipliers with different accumulators (Berkeley HardFloat Release 1,
// recoded formats, RNE fixed, full IEEE: subnormals, Inf, NaN).  HardFloat itself is not copied here,
// scripts/hw/fetch_hw_deps.sh downloads it; scripts/hw/array/flow.py patches addRecFN (bf16 bug).
//
// Accumulator semantics of one MAC step (a, b bf16, product exact):   s <- RNE_acc(s + a*b)
// (a single rounding of the exact value).  VAR selects the accumulator:
//   VAR 0  FP32   e8m23  : exact bf16xbf16->fp32 product + fp32 adder (addRecFN 8/24)
//   VAR 1  BF16   e8m7   : fused  mulAddRecFNToRaw(8,8) + round to (8,8)
//   VAR 2  FP16   e5m10  : fused  mulAddRecFNToRaw(8,11) (bf16 range for the product) + round to (5,11)
//   VAR 3  E8M13  e8m13  : fused  mulAddRecFNToRaw(8,14) (operands zero-padded) + round to (8,14)
//   VAR 4  KAHAN  bf16 s + bf16 c : y=RN(ab-c) (fused), t=RN(s+y), z=RN(t-s), c'=RN(z-y)
//   VAR 5  CHUNK  in-array chain = VAR 1 (psum enters each K-tile as +0); FP32 adder per column at the
//                 bottom edge: acc32 <- RN32(acc32 + psum_bf16)          (chunk C = R = array depth)
//   VAR 11/12/13  "separate" implementations of VAR 1/2/3: exact product rounded to (8,16) (exact for
//                 normal-range products) + addRecFNToRaw(8,16) + one rounding to the target format.
// Recoded widths: bf16/fp16 17 b, e8m13 23 b, fp32 33 b.
`include "HardFloat_consts.vi"
`define RNE 3'b000
`define CTL `flControl_tininessAfterRounding

// ------------------------------------------------------------------------------------------------
// one accumulation step  s_out = RN_(AE,AS)(s_in + a*b), a/b recoded bf16
// ------------------------------------------------------------------------------------------------
module acc_step #(parameter AE = 8, parameter AS = 8, parameter IMPL = 0)   // IMPL 0 fused, 1 separate
    (input [16:0] a, input [16:0] b, input [AE+AS:0] s, output [AE+AS:0] y);
    wire [4:0] f0, f1, f2;
    generate
    if (AE == 8 && AS == 24) begin : g_fp32
        // exact product in fp32 (8x8 significand multiplier) + fp32 adder
        wire inv, isNaN, isInf, isZero, sgn; wire signed [9:0] sExp; wire [15:0] sig; wire [32:0] p;
        mulRecFNToFullRaw#(8, 8) m (`CTL, a, b, inv, isNaN, isInf, isZero, sgn, sExp, sig);
        roundAnyRawFNToRecFN#(8, 15, 8, 24, 0) r (`CTL, inv, 1'b0, isNaN, isInf, isZero, sgn, sExp, sig, `RNE, p, f0);
        addRecFN#(8, 24) ad (`CTL, 1'b0, s, p, `RNE, y, f1);
    end else if (IMPL == 0) begin : g_fused
        localparam P = (AS > 8) ? AS : 8;               // working precision of the fused unit
        wire [8+P:0] aw = {a, {(P-8){1'b0}}};            // zero-pad fraction (exact)
        wire [8+P:0] bw = {b, {(P-8){1'b0}}};
        wire [8+P:0] cw;
        if (AE == 8) begin : g_c8
            assign cw = s;                               // AS == P
        end else begin : g_cwide                         // fp16 -> (8,11), exact
            recFNToRecFN#(AE, AS, 8, P) cv (`CTL, s, `RNE, cw, f2);
        end
        wire inv, isNaN, isInf, isZero, sgn; wire signed [9:0] sExp; wire [P+2:0] sig;
        mulAddRecFNToRaw#(8, P) fma (`CTL, 2'b00, aw, bw, cw, `RNE, inv, isNaN, isInf, isZero, sgn, sExp, sig);
        roundAnyRawFNToRecFN#(8, P+2, AE, AS, 0) r (`CTL, inv, 1'b0, isNaN, isInf, isZero, sgn, sExp, sig, `RNE, y, f0);
    end else begin : g_sep
        // exact product as (8,16) + (8,16) adder to raw + single rounding to (AE,AS)
        wire inv, isNaN, isInf, isZero, sgn; wire signed [9:0] sExp; wire [15:0] sig; wire [24:0] p, sw;
        mulRecFNToFullRaw#(8, 8) m (`CTL, a, b, inv, isNaN, isInf, isZero, sgn, sExp, sig);
        roundAnyRawFNToRecFN#(8, 15, 8, 16, 0) r (`CTL, inv, 1'b0, isNaN, isInf, isZero, sgn, sExp, sig, `RNE, p, f0);
        recFNToRecFN#(AE, AS, 8, 16) cv (`CTL, s, `RNE, sw, f2);
        wire inv2, isNaN2, isInf2, isZero2, sgn2; wire signed [9:0] sExp2; wire [18:0] sig2;
        addRecFNToRaw#(8, 16) ad (`CTL, 1'b0, sw, p, `RNE, inv2, isNaN2, isInf2, isZero2, sgn2, sExp2, sig2);
        roundAnyRawFNToRecFN#(8, 18, AE, AS, 0) r2 (`CTL, inv2, 1'b0, isNaN2, isInf2, isZero2, sgn2, sExp2, sig2, `RNE, y, f1);
    end
    endgenerate
endmodule

// Kahan-compensated bf16 step: {s,c} -> {t,c'}
module kahan_step (input [16:0] a, input [16:0] b, input [16:0] s, input [16:0] c,
                   output [16:0] t, output [16:0] cn);
    wire [4:0] f0, f1, f2, f3;
    wire [16:0] yv, z;
    wire inv, isNaN, isInf, isZero, sgn; wire signed [9:0] sExp; wire [10:0] sig;
    mulAddRecFNToRaw#(8, 8) fma (`CTL, 2'b01, a, b, c, `RNE, inv, isNaN, isInf, isZero, sgn, sExp, sig); // a*b - c
    roundAnyRawFNToRecFN#(8, 10, 8, 8, 0) r (`CTL, inv, 1'b0, isNaN, isInf, isZero, sgn, sExp, sig, `RNE, yv, f0);
    addRecFN#(8, 8) a1 (`CTL, 1'b0, s, yv, `RNE, t, f1);    // t  = s + y
    addRecFN#(8, 8) a2 (`CTL, 1'b1, t, s, `RNE, z, f2);     // z  = t - s
    addRecFN#(8, 8) a3 (`CTL, 1'b1, z, yv, `RNE, cn, f3);   // c' = z - y
endmodule

// ------------------------------------------------------------------------------------------------
// weight-stationary PE: weight register (shift-in from north while wload), a forwarded east,
// partial sum (SW bits, recoded) enters from north and leaves south (registered).
// ------------------------------------------------------------------------------------------------
(* keep_hierarchy *)
module pe_ws #(parameter VAR = 0, parameter SW = 33)
    (input clk, input wload, input [16:0] w_in, input [16:0] a_in, input [SW-1:0] s_in,
     output reg [16:0] w, output reg [16:0] a_out, output reg [SW-1:0] s_out);
    wire [SW-1:0] s_nx;
    generate
    if (VAR == 0)       begin : v acc_step #(8, 24, 0) u (a_in, w, s_in, s_nx); end
    else if (VAR == 1 || VAR == 5) begin : v acc_step #(8, 8, 0) u (a_in, w, s_in, s_nx); end
    else if (VAR == 2)  begin : v acc_step #(5, 11, 0) u (a_in, w, s_in, s_nx); end
    else if (VAR == 3)  begin : v acc_step #(8, 14, 0) u (a_in, w, s_in, s_nx); end
    else if (VAR == 11) begin : v acc_step #(8, 8, 1) u (a_in, w, s_in, s_nx); end
    else if (VAR == 12) begin : v acc_step #(5, 11, 1) u (a_in, w, s_in, s_nx); end
    else if (VAR == 13) begin : v acc_step #(8, 14, 1) u (a_in, w, s_in, s_nx); end
    else if (VAR == 4)  begin : v kahan_step u (a_in, w, s_in[33:17], s_in[16:0], s_nx[33:17], s_nx[16:0]); end
    endgenerate
    always @(posedge clk) begin
        if (wload) w <= w_in;
        a_out <= a_in;
        s_out <= s_nx;
    end
endmodule

// output-stationary PE: a east, b south, accumulator stays; clr starts a new output (s <- 0 + a*b).
// (drain path not modelled: the accumulator is read out by a separate shift chain in a real design)
module pe_os #(parameter VAR = 0, parameter SW = 33)
    (input clk, input clr, input [16:0] a_in, input [16:0] b_in,
     output reg [16:0] a_out, output reg [16:0] b_out, output reg [SW-1:0] acc);
    wire [SW-1:0] s_in = clr ? {SW{1'b0}} : acc;
    wire [SW-1:0] s_nx;
    generate
    if (VAR == 0)       begin : v acc_step #(8, 24, 0) u (a_in, b_in, s_in, s_nx); end
    else if (VAR == 1 || VAR == 5) begin : v acc_step #(8, 8, 0) u (a_in, b_in, s_in, s_nx); end
    else if (VAR == 2)  begin : v acc_step #(5, 11, 0) u (a_in, b_in, s_in, s_nx); end
    else if (VAR == 3)  begin : v acc_step #(8, 14, 0) u (a_in, b_in, s_in, s_nx); end
    else if (VAR == 4)  begin : v kahan_step u (a_in, b_in, s_in[33:17], s_in[16:0], s_nx[33:17], s_nx[16:0]); end
    endgenerate
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
        acc <= s_nx;
    end
endmodule

// bottom-edge unit of the chunked variant: acc32_out = RN32(acc32_in + psum_bf16), registered
(* keep_hierarchy *)
module edge_chunk (input clk, input [16:0] ps, input [31:0] acc_in, output reg [31:0] acc_out);
    wire [32:0] ra, sum; wire [31:0] o; wire [4:0] f;
    fNToRecFN#(8, 24) ci (acc_in, ra);
    addRecFN#(8, 24) ad (`CTL, 1'b0, ra, {ps, 16'b0}, `RNE, sum, f);   // bf16 -> fp32 is exact
    recFNToFN#(8, 24) co (sum, o);
    always @(posedge clk) acc_out <= o;
endmodule

// ------------------------------------------------------------------------------------------------
// R x C weight-stationary array.  I/O in standard IEEE encodings:
//   a_in  : R bf16 (row r must be skewed by r cycles outside), w_in: C bf16 (shifted in during wload)
//   ps_in : C partial sums in the accumulator format (column c skewed by c cycles); the previous
//           K-tile's result (sequential semantics across K-tiles; accumulator SRAM outside).  Unused
//           for VAR 5 (each K-tile starts from +0 = new chunk).
//   ps_out: C results of the column chains (VAR 5: fp32 results of the edge adders, acc_in = old value)
// ------------------------------------------------------------------------------------------------
module ws_array #(parameter R = 16, parameter C = 16, parameter VAR = 0,
        parameter AE = (VAR == 2 || VAR == 12) ? 5 : 8,
        parameter AS = (VAR == 0) ? 24 : (VAR == 2 || VAR == 12) ? 11 : (VAR == 3 || VAR == 13) ? 14 : 8,
        parameter SW = (VAR == 4) ? 34 : AE + AS + 1,           // recoded psum bus width
        parameter IW = (VAR == 4) ? 32 : AE + AS,               // IEEE width at the edges
        parameter OW = (VAR == 5) ? 32 : IW)
    (input clk, input wload, input [R*16-1:0] a_in, input [C*16-1:0] w_in,
     input [C*IW-1:0] ps_in, input [C*32-1:0] acc_in, output [C*OW-1:0] ps_out);
    wire [16:0] aw [0:R][0:C];
    wire [16:0] ww [0:R][0:C];
    wire [SW-1:0] sw [0:R][0:C];
    genvar r, c;
    generate
    for (r = 0; r < R; r = r + 1) begin : west
        fNToRecFN#(8, 8) cv (a_in[r*16 +: 16], aw[r][0]);
    end
    for (c = 0; c < C; c = c + 1) begin : top
        fNToRecFN#(8, 8) cw (w_in[c*16 +: 16], ww[0][c]);
        if (VAR == 5) begin : z
            assign sw[0][c] = {SW{1'b0}};
        end else if (VAR == 4) begin : k
            fNToRecFN#(8, 8) cs (ps_in[c*IW + 16 +: 16], sw[0][c][33:17]);
            fNToRecFN#(8, 8) cc (ps_in[c*IW +: 16], sw[0][c][16:0]);
        end else begin : s
            fNToRecFN#(AE, AS) cs (ps_in[c*IW +: IW], sw[0][c]);
        end
    end
    for (r = 0; r < R; r = r + 1) begin : row
        for (c = 0; c < C; c = c + 1) begin : col
            pe_ws #(VAR, SW) pe (clk, wload, ww[r][c], aw[r][c], sw[r][c], ww[r+1][c], aw[r][c+1], sw[r+1][c]);
        end
    end
    for (c = 0; c < C; c = c + 1) begin : bottom
        if (VAR == 5) begin : e
            edge_chunk u (clk, sw[R][c], acc_in[c*32 +: 32], ps_out[c*32 +: 32]);
        end else if (VAR == 4) begin : k
            recFNToFN#(8, 8) os (sw[R][c][33:17], ps_out[c*OW + 16 +: 16]);
            recFNToFN#(8, 8) oc (sw[R][c][16:0], ps_out[c*OW +: 16]);
        end else begin : o
            recFNToFN#(AE, AS) os (sw[R][c], ps_out[c*OW +: OW]);
        end
    end
    endgenerate
endmodule


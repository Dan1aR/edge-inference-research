// E6 hardware study: BF16-input MAC units with FP32 vs BF16 accumulation.
// Built on Berkeley HardFloat Release 1 (recoded FP format, expWidth/sigWidth parameters;
// http://www.jhauser.us/arithmetic/HardFloat.html, BSD-3 licence).  HardFloat sources are NOT
// copied into this repo; scripts/hw/fetch_hw_deps.sh downloads them.
//
// Formats (HardFloat "recoded" = 1 extra exponent bit):
//   bf16  : expWidth=8, sigWidth=8   -> 17-bit recoded word
//   fp32  : expWidth=8, sigWidth=24  -> 33-bit recoded word
// Because expWidth is identical, a recoded bf16 is widened to a recoded fp32 *exactly* by
// appending 16 zero fraction bits.  A bf16 x bf16 product has a 16-bit significand and is
// therefore exact in fp32 (unless it leaves the fp32 exponent range -> handled by rounding).
//
// Rounding mode: fixed round-to-nearest-even, tininess after rounding (typical for an
// inference accelerator).  Full IEEE semantics otherwise (subnormals, Inf/NaN, flags).
`include "HardFloat_consts.vi"

`define RNE 3'b000
`define CTL `flControl_tininessAfterRounding

// ----------------------------------------------------------------------------------------
// Reference arithmetic blocks (combinational)
// ----------------------------------------------------------------------------------------
module add_fp32 (input [32:0] a, input [32:0] b, output [32:0] y);
    wire [4:0] f;
    addRecFN#(8, 24) u (`CTL, 1'b0, a, b, `RNE, y, f);
endmodule

module add_bf16 (input [16:0] a, input [16:0] b, output [16:0] y);
    wire [4:0] f;
    addRecFN#(8, 8) u (`CTL, 1'b0, a, b, `RNE, y, f);
endmodule

module mul_bf16 (input [16:0] a, input [16:0] b, output [16:0] y);   // product rounded to bf16
    wire [4:0] f;
    mulRecFN#(8, 8) u (`CTL, a, b, `RNE, y, f);
endmodule

module mul_fp32 (input [32:0] a, input [32:0] b, output [32:0] y);   // full fp32 multiplier
    wire [4:0] f;
    mulRecFN#(8, 24) u (`CTL, a, b, `RNE, y, f);
endmodule

// bf16 x bf16 -> fp32, exact product (8x8 significand multiplier + exponent add + range check)
module mul_bf16_exact_fp32 (input [16:0] a, input [16:0] b, output [32:0] y);
    wire inv, isNaN, isInf, isZero, sgn;
    wire signed [9:0] sExp;
    wire [15:0] sig;
    wire [4:0] f;
    mulRecFNToFullRaw#(8, 8) m (`CTL, a, b, inv, isNaN, isInf, isZero, sgn, sExp, sig);
    roundAnyRawFNToRecFN#(8, 15, 8, 24, 0)
        r (`CTL, inv, 1'b0, isNaN, isInf, isZero, sgn, sExp, sig, `RNE, y, f);
endmodule

// fused multiply-add, bf16 operands widened to fp32, fp32 addend/result (one rounding)
module fma_fp32_bf16in (input [16:0] a, input [16:0] b, input [32:0] c, output [32:0] y);
    wire [4:0] f;
    mulAddRecFN#(8, 24) u (`CTL, 2'b00, {a, 16'b0}, {b, 16'b0}, c, `RNE, y, f);
endmodule

// fused multiply-add fully in bf16 (one rounding to bf16)
module fma_bf16 (input [16:0] a, input [16:0] b, input [16:0] c, output [16:0] y);
    wire [4:0] f;
    mulAddRecFN#(8, 8) u (`CTL, 2'b00, a, b, c, `RNE, y, f);
endmodule

// ----------------------------------------------------------------------------------------
// Systolic-array processing elements (output-stationary): operand forwarding registers
// (a -> east, b -> south), accumulator register, 'clr' starts a new output.
// ----------------------------------------------------------------------------------------
// FP32 accumulation, separate exact multiplier + fp32 adder  ("BF16-in / FP32-accum")
module pe_fp32acc_sep (input clk, input clr, input [16:0] a, input [16:0] b,
                       output reg [16:0] a_o, output reg [16:0] b_o, output reg [32:0] acc);
    wire [32:0] p, s;
    mul_bf16_exact_fp32 m (a, b, p);
    add_fp32 ad (clr ? 33'd0 : acc, p, s);
    always @(posedge clk) begin a_o <= a; b_o <= b; acc <= s; end
endmodule

// BF16 accumulation, product rounded to bf16 + bf16 adder (= numerics of src/bf16_accum.py)
module pe_bf16acc_sep (input clk, input clr, input [16:0] a, input [16:0] b,
                       output reg [16:0] a_o, output reg [16:0] b_o, output reg [16:0] acc);
    wire [16:0] p, s;
    mul_bf16 m (a, b, p);
    add_bf16 ad (clr ? 17'd0 : acc, p, s);
    always @(posedge clk) begin a_o <= a; b_o <= b; acc <= s; end
endmodule

// FP32 accumulation, fused (bf16 operands, fp32 accumulator)
module pe_fp32acc_fma (input clk, input clr, input [16:0] a, input [16:0] b,
                       output reg [16:0] a_o, output reg [16:0] b_o, output reg [32:0] acc);
    wire [32:0] s;
    fma_fp32_bf16in u (a, b, clr ? 33'd0 : acc, s);
    always @(posedge clk) begin a_o <= a; b_o <= b; acc <= s; end
endmodule

// BF16 accumulation, fused
module pe_bf16acc_fma (input clk, input clr, input [16:0] a, input [16:0] b,
                       output reg [16:0] a_o, output reg [16:0] b_o, output reg [16:0] acc);
    wire [16:0] s;
    fma_bf16 u (a, b, clr ? 17'd0 : acc, s);
    always @(posedge clk) begin a_o <= a; b_o <= b; acc <= s; end
endmodule

// ----------------------------------------------------------------------------------------
// Functional-test wrappers (standard IEEE encodings in/out), used only for gate-level
// self-check against torch/numpy in run_synth.py
// ----------------------------------------------------------------------------------------
module t_fp32acc_sep (input [15:0] a, input [15:0] b, input [31:0] c, output [31:0] y);
    wire [16:0] ra, rb; wire [32:0] rc, p, s;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    fNToRecFN#(8, 24) cc (c, rc);
    mul_bf16_exact_fp32 m (ra, rb, p);
    add_fp32 ad (rc, p, s);
    recFNToFN#(8, 24) co (s, y);
endmodule

module t_bf16acc_sep (input [15:0] a, input [15:0] b, input [15:0] c, output [15:0] y);
    wire [16:0] ra, rb, rc, p, s;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    fNToRecFN#(8, 8) cc (c, rc);
    mul_bf16 m (ra, rb, p);
    add_bf16 ad (rc, p, s);
    recFNToFN#(8, 8) co (s, y);
endmodule

module t_fp32acc_fma (input [15:0] a, input [15:0] b, input [31:0] c, output [31:0] y);
    wire [16:0] ra, rb; wire [32:0] rc, s;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    fNToRecFN#(8, 24) cc (c, rc);
    fma_fp32_bf16in u (ra, rb, rc, s);
    recFNToFN#(8, 24) co (s, y);
endmodule

module t_bf16acc_fma (input [15:0] a, input [15:0] b, input [15:0] c, output [15:0] y);
    wire [16:0] ra, rb, rc, s;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    fNToRecFN#(8, 8) cc (c, rc);
    fma_bf16 u (ra, rb, rc, s);
    recFNToFN#(8, 8) co (s, y);
endmodule

module t_add_bf16 (input [15:0] a, input [15:0] b, input [15:0] c, output [15:0] y);
    wire [16:0] ra, rc, s;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cc (c, rc);
    add_bf16 ad (ra, rc, s);
    recFNToFN#(8, 8) co (s, y);
endmodule

module t_mul_bf16 (input [15:0] a, input [15:0] b, input [15:0] c, output [15:0] y);
    wire [16:0] ra, rb, p;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    mul_bf16 m (ra, rb, p);
    recFNToFN#(8, 8) co (p, y);
endmodule

// x = {a, b} is an arbitrary fp32 operand (full 24-bit significand)
module t_add_fp32 (input [15:0] a, input [15:0] b, input [31:0] c, output [31:0] y);
    wire [32:0] rx, rc, s;
    fNToRecFN#(8, 24) cx ({a, b}, rx);
    fNToRecFN#(8, 24) cc (c, rc);
    add_fp32 ad (rx, rc, s);
    recFNToFN#(8, 24) co (s, y);
endmodule

module t_mul_bf16_exact (input [15:0] a, input [15:0] b, input [31:0] c, output [31:0] y);
    wire [16:0] ra, rb; wire [32:0] p;
    fNToRecFN#(8, 8) ca (a, ra);
    fNToRecFN#(8, 8) cb (b, rb);
    mul_bf16_exact_fp32 m (ra, rb, p);
    recFNToFN#(8, 24) co (p, y);
endmodule

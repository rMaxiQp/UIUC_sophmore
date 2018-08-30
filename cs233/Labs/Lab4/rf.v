// register: A register which may be reset to an arbirary value
//
// q      (output) - Current value of register
// d      (input)  - Next value of register
// clk    (input)  - Clock (positive edge-sensitive)
// enable (input)  - Load new value? (yes = 1, no = 0)
// reset  (input)  - Asynchronous reset    (reset = 1)
//
module register(q, d, clk, enable, reset);

    parameter
        width = 32,
        reset_value = 0;

    output [(width-1):0] q;
    reg    [(width-1):0] q;
    input  [(width-1):0] d;
    input  clk, enable, reset;

    always@(reset)
      if (reset == 1'b1)
        q <= reset_value;

    always@(posedge clk)
      if ((reset == 1'b0) && (enable == 1'b1))
        q <= d;

endmodule // register

module decoder2 (out, in, enable);
    input     in;
    input     enable;
    output [1:0] out;

    and a0(out[0], enable, ~in);
    and a1(out[1], enable, in);
endmodule // decoder2

module decoder4 (out, in, enable);
    input [1:0]    in;
    input     enable;
    output [3:0]   out;
    wire [1:0]    w_enable;

    decoder2 d1(out[1:0], in[0], w_enable[0]);
    decoder2 d2(out[3:2], in[0], w_enable[1]);
    decoder2 d3(w_enable[1:0], in[1], enable);
endmodule // decoder4

module decoder8 (out, in, enable);
    input [2:0]    in;
    input     enable;
    output [7:0]   out;
    wire [1:0]    w_enable;

    decoder4 d1(out[3:0], in[1:0], w_enable[0]);
    decoder4 d2(out[7:4], in[1:0], w_enable[1]);
    decoder2 d3(w_enable[1:0], in[2], enable);


endmodule // decoder8

module decoder16 (out, in, enable);
    input [3:0]    in;
    input     enable;
    output [15:0]  out;
    wire [1:0]    w_enable;

    decoder8 d1(out[15:8], in[2:0], w_enable[1]);
    decoder8 d2(out[7:0], in[2:0], w_enable[0]);
    decoder2 d3(w_enable[1:0], in[3], enable);

endmodule // decoder16

module decoder32 (out, in, enable);
    input [4:0]    in;
    input     enable;
    output [31:0]  out;
    wire [1:0]    w_enable;

    decoder16 d1(out[31:16], in[3:0], w_enable[1]);
    decoder16 d2(out[15:0], in[3:0], w_enable[0]);
    decoder2 d3(w_enable[1:0], in[4], enable);

endmodule // decoder32

module mips_regfile (rd1_data, rd2_data, rd1_regnum, rd2_regnum,
             wr_regnum, wr_data, writeenable,
             clock, reset);

    output [31:0]  rd1_data, rd2_data;
    input   [4:0]  rd1_regnum, rd2_regnum, wr_regnum;
    input  [31:0]  wr_data;
    input          writeenable, clock, reset;
    wire  [31:0] data;
    wire  [31:0] w[31:0];

    decoder32 d1(data, wr_regnum, writeenable);

    assign w[0] = 0;

    register r1(w[1], wr_data, clock, data[1], reset);
    register r2(w[2], wr_data, clock, data[2], reset);
    register r3(w[3], wr_data, clock, data[3], reset);
    register r4(w[4], wr_data, clock, data[4], reset);
    register r5(w[5], wr_data, clock, data[5], reset);
    register r6(w[6], wr_data, clock, data[6], reset);
    register r7(w[7], wr_data, clock, data[7], reset);
    register r8(w[8], wr_data, clock, data[8], reset);
    register r9(w[9], wr_data, clock, data[9], reset);
    register r10(w[10], wr_data, clock, data[10], reset);
    register r11(w[11], wr_data, clock, data[11], reset);
    register r12(w[12], wr_data, clock, data[12], reset);
    register r13(w[13], wr_data, clock, data[13], reset);
    register r14(w[14], wr_data, clock, data[14], reset);
    register r15(w[15], wr_data, clock, data[15], reset);
    register r16(w[16], wr_data, clock, data[16], reset);
    register r17(w[17], wr_data, clock, data[17], reset);
    register r18(w[18], wr_data, clock, data[18], reset);
    register r19(w[19], wr_data, clock, data[19], reset);
    register r20(w[20], wr_data, clock, data[20], reset);
    register r21(w[21], wr_data, clock, data[21], reset);
    register r22(w[22], wr_data, clock, data[22], reset);
    register r23(w[23], wr_data, clock, data[23], reset);
    register r24(w[24], wr_data, clock, data[24], reset);
    register r25(w[25], wr_data, clock, data[25], reset);
    register r26(w[26], wr_data, clock, data[26], reset);
    register r27(w[27], wr_data, clock, data[27], reset);
    register r28(w[28], wr_data, clock, data[28], reset);
    register r29(w[29], wr_data, clock, data[29], reset);
    register r30(w[30], wr_data, clock, data[30], reset);
    register r31(w[31], wr_data, clock, data[31], reset);


    mux32v m1(rd1_data, w[0], w[1], w[2], w[3], w[4], w[5],
    w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14],
     w[15], w[16], w[17], w[18], w[19], w[20], w[21], w[22],
    w[23], w[24], w[25], w[26], w[27], w[28], w[29], w[30], w[31], rd1_regnum);

    mux32v m2(rd2_data, w[0], w[1], w[2], w[3], w[4], w[5],
    w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14],
     w[15], w[16], w[17], w[18], w[19], w[20], w[21], w[22],
    w[23], w[24], w[25], w[26], w[27], w[28], w[29], w[30], w[31], rd2_regnum);

endmodule // mips_regfile

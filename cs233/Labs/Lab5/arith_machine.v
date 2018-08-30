// arith_machine: execute a series of arithmetic instructions from an instruction cache
//
// except (output) - set to 1 when an unrecognized instruction is to be executed.
// clock  (input)  - the clock signal
// reset  (input)  - set to 1 to set all registers to zero, set to 0 for normal execution.

module arith_machine(except, clock, reset);
    output      except;
    input       clock, reset;

    wire [31:0] inst, rdData, rtData;
    wire [31:0] PC, nextPC, rsData, w_B;
    wire [31:0] imm = {{16{inst[15]}}, inst[15:0]};
    wire [4:0] rdNum;
    wire [2:0] alu_op;
    wire writeenable, alu_src2, rd_src;


    register #(32) PC_reg(PC, nextPC, clock, 1, reset);


    instruction_memory im(inst ,PC[31:2]);


    regfile rf (rsData, rtData, inst[25:21], inst[20:16], rdNum, rdData, writeenable, clock, reset);

    alu32 a1(nextPC, , , , PC, 32'h4, `ALU_ADD);
    alu32 a2(rdData, , , , rsData, w_B, alu_op);

    mux2v #(5) m2(rdNum, inst[15:11], inst[20:16], rd_src);
    mux2v #(32) m3(w_B, rtData, imm, alu_src2);

    mips_decode md1(alu_op, writeenable, rd_src, alu_src2, except, inst[31:26], inst[5:0]);

endmodule // arith_machine

// full_machine: execute a series of MIPS instructions from an instruction cache
//
// except (output) - set to 1 when an unrecognized instruction is to be executed.
// clock   (input) - the clock signal
// reset   (input) - set to 1 to set all registers to zero, set to 0 for normal execution.

module full_machine(except, clock, reset);
    output      except;
    input       clock, reset;

    wire [31:0] inst, rdData, rtData, PC, nextPC, rsData, data_out;
    wire [31:0] w_B, w_j, w_PC, w_mr, w_slt, w_out, w_byte, addr , w_rd, mRd;
    wire [7:0]  w_data;
    wire [31:0] imm = {{16{inst[15]}}, inst[15:0]};
    wire [31:0] jump = {PC[31:28],inst[25:0],2'b0};
    wire [31:0] branch = {imm[29:0],2'b0};
    wire [4:0] rdNum, rd, rt, rs;
    wire [2:0] alu_op;
    wire [1:0] control_type;
    wire negative [3:0], zero [3:0] , overflow [3:0];
    wire writeenable, alu_src2, rd_src, mem_read, word_we, lui, slt, addm, byte_we, byte_load;

    assign rd = inst[15:11];
    assign rt = inst[20:16];
    assign rs = inst[25:21];

    register #(32) PC_reg(PC, nextPC, clock, 1, reset);

    instruction_memory im(inst ,PC[31:2]);

    regfile rf (rsData, rtData, rs, rt, rdNum,
                rdData, writeenable, clock, reset);

    alu32 a1(w_PC, overflow[0], zero[0], negative[0], PC, 32'h4, `ALU_ADD);
    alu32 a2(w_out, overflow[1], zero[1], negative[1], rsData, w_B, alu_op);
    alu32 a3(w_j, overflow[2], zero[2], negative[2], w_PC, branch, `ALU_ADD);
    alu32 a4(mRd, overflow[3], zero[3], negative[3], rtData , w_mr, `ALU_ADD);

    mux2v #(5) m2(rdNum, rd, rt, rd_src);
    mux2v #(32) m3(w_B, rtData, imm, alu_src2);
    mux2v #(32) m4(rdData, w_rd, {inst[15:0], 16'h0}, lui);
    mux2v #(32) m5(w_slt, w_out, {31'b0, ((overflow[1]& ~ negative[1])|(negative[1]& ~overflow[1]))}, slt);
    mux2v #(32) m6(w_mr, w_slt, w_byte, mem_read);
    mux2v #(32) m7(w_byte, data_out, {24'b0, w_data}, byte_load);
    mux2v #(32) m8(addr, w_out, rsData, addm);
    mux2v #(32) m9(w_rd, w_mr, mRd, addm);

    mux4v #(32) m45(nextPC, w_PC, w_j, jump, rsData, control_type);
    mux4v #(8) m55(w_data, data_out[7:0], data_out[15:8], data_out[23:16], data_out[31:24], w_out[1:0]);

    mips_decode md1(alu_op, writeenable, rd_src, alu_src2, except, control_type,
                       mem_read, word_we, byte_we, byte_load, lui, slt, addm,
                       inst[31:26], inst[5:0], zero[1]);

    data_mem dm1(data_out, addr, rtData, word_we, byte_we, clock, reset);
endmodule // full_machine

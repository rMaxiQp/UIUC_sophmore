module pipelined_machine(clk, reset);
    input        clk, reset;

    wire [31:0]  PC;
    wire [31:2]  next_PC, PC_plus4, PC_target;
    wire [31:0]  inst;

    wire [31:0]  imm = {{ 16{inst[15]} }, inst[15:0] };  // sign-extended immediate
    wire [4:0]   rs = inst[25:21];
    wire [4:0]   rt = inst[20:16];
    wire [4:0]   rd = inst[15:11];
    wire [5:0]   opcode = inst[31:26];
    wire [5:0]   funct = inst[5:0];

    wire [4:0]   wr_regnum;
    wire [2:0]   ALUOp;

    wire         RegWrite, BEQ, ALUSrc, MemRead, MemWrite, MemToReg, RegDst;
    wire         PCSrc, zero;
    wire [31:0]  rd1_data, rd2_data, B_data, alu_out_data, load_data, wr_data;

    // START OF NEW WIRES
    wire [31:0]  PC_S_plus4, PC_inst, alu_out_data_2, rd2_data_2, rd_data, rt_data;
    wire [9:0]   prev_decode, decode_out;
    wire [4:0]   prev_regnum;
    wire         mw, mr, mtReg, rw, forwardA, forwardB, stall, flush;
    // END OF NEW WIRES

    //PIPELINING
    // IF/DE << start
    register #(30, 30'h100000) pc_hold(PC_S_plus4[31:2], PC_plus4[31:2], clk, ~stall, flush);
    register #(32) inst_hold(inst, PC_inst, clk, ~stall, flush);
    // IF/DE << end

    // DE/MW << start
    register #(32) alu_hold(alu_out_data_2 ,alu_out_data, clk, 1'b1, flush);
    register #(1) m_write(mw, MemWrite, clk, 1'b1, flush);
    register #(1) m_read(mr, MemRead, clk, 1'b1, flush);
    register #(1) m_to_reg(mtReg, MemToReg, clk, 1'b1, flush);
    register #(1) reg_write(rw, RegWrite, clk, 1'b1, flush);
    register #(32) write_hold(rd2_data_2, rt_data, clk, 1'b1, flush);
    register #(5)  write_regnum_hold(wr_regnum, prev_regnum, clk, 1'b1, flush);
    // DE/MW << end

    //FORWARDING
    assign forwardA = rw & (wr_regnum == rs) & (wr_regnum != 5'b0);
    assign forwardB = rw & (wr_regnum == rt) & (wr_regnum != 5'b0);
    mux2v #(32) alu_rd1(rd_data, rd1_data, alu_out_data_2, forwardA);
    mux2v #(32) alu_rd2(rt_data, rd2_data, alu_out_data_2, forwardB);

    //STALLING
    assign stall = (wr_regnum == rs | wr_regnum == rt) & mr;

    //FLUSHING
    assign flush = reset | PCSrc;


    // DO NOT comment out or rename this module
    // or the test bench will break
    register #(30, 30'h100000) PC_reg(PC[31:2], next_PC[31:2], clk, /* enable */~stall, reset);

    assign PC[1:0] = 2'b0;  // bottom bits hard coded to 00
    adder30 next_PC_adder(PC_plus4, PC[31:2], 30'h1);
    adder30 target_PC_adder(PC_target[31:2], PC_S_plus4[31:2], imm[29:0]);
    mux2v #(30) branch_mux(next_PC, PC_plus4, PC_target, PCSrc);
    assign PCSrc = BEQ & zero;

    // DO NOT comment out or rename this module
    // or the test bench will break
    instruction_memory imem(PC_inst, PC[31:2]);

    mips_decode decode(ALUOp, RegWrite, BEQ, ALUSrc, MemRead, MemWrite, MemToReg, RegDst,
                      opcode, funct);

    // DO NOT comment out or rename this module
    // or the test bench will break
    regfile rf (rd1_data, rd2_data,
               rs, rt, wr_regnum, wr_data,
               rw, clk, reset);

    mux2v #(32) imm_mux(B_data, rt_data, imm, ALUSrc);
    alu32 alu(alu_out_data, zero, ALUOp, rd_data, B_data);

    // DO NOT comment out or rename this module
    // or the test bench will break
    data_mem data_memory(load_data, alu_out_data_2, rd2_data_2, mr, mw, clk, reset);

    mux2v #(32) wb_mux(wr_data, alu_out_data_2, load_data, mtReg);
    mux2v #(5) rd_mux(prev_regnum, rt, rd, RegDst);

endmodule // pipelined_machine

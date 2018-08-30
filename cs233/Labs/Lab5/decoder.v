// mips_decode: a decoder for MIPS arithmetic instructions
//
// alu_op      (output) - control signal to be sent to the ALU
// writeenable (output) - should a new value be captured by the register file
// rd_src      (output) - should the destination register be rd (0) or rt (1)
// alu_src2    (output) - should the 2nd ALU source be a register (0) or an immediate (1)
// except      (output) - set to 1 when the opcode/funct combination is unrecognized
// opcode      (input)  - the opcode field from the instruction
// funct       (input)  - the function field from the instruction
//

module mips_decode(alu_op, writeenable, rd_src, alu_src2, except, opcode, funct);
    output [2:0] alu_op;
    output       writeenable, rd_src, alu_src2, except;
    input  [5:0] opcode, funct;
    wire add, sub, and_, or_, nor_, xor_, addi, andi, ori, xori;

    assign add = opcode == 0 & funct == 6'h20;
    assign sub = opcode == 0 & funct == 6'h22;
    assign and_ = opcode == 0 & funct == 6'h24;
    assign or_ = opcode == 0 & funct == 6'h25;
    assign nor_ = opcode == 0 & funct == 6'h27;
    assign xor_ = opcode == 0 & funct == 6'h26;
    assign addi = opcode == 6'h8;
    assign andi = opcode == 6'hc;
    assign ori = opcode == 6'hd;
    assign xori = opcode == 6'he;

    assign rd_src = addi | andi | ori | xori;
    assign alu_src2 = addi | andi | ori | xori ;
    assign writeenable = add | sub | and_ | or_ | nor_ | xor_ | addi | andi | ori | xori;
    assign alu_op[2] = and_ | or_ | nor_ | xor_ | andi | ori | xori;
    assign alu_op[1] = add | sub | nor_ | xor_ | addi | xori;
    assign alu_op[0] = sub | or_ | xor_ | ori | xori;
    assign except = ~writeenable;


endmodule // mips_decode

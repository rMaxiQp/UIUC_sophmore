module decoder_test;
    reg [5:0] opcode, funct;

    initial begin
        $dumpfile("decoder.vcd");
        $dumpvars(0, decoder_test);

             opcode = `OP_OTHER0; funct = `OP0_ADD; // try addition
        # 10 opcode = `OP_OTHER0; funct = `OP0_SUB; // try subtraction
        # 10 opcode = `OP_OTHER0; funct = `OP0_AND;
        # 10 opcode = `OP_OTHER0; funct = `OP0_OR;
        # 10 opcode = `OP_OTHER0; funct = `OP0_NOR;
        # 10 opcode = `OP_OTHER0; funct = `OP0_XOR;
        # 10 opcode = `OP_OTHER0; funct = 0;
        # 10 $finish;
    end

    // use gtkwave to test correctness
    wire [2:0] alu_op;
    wire       writeenable, rd_src, alu_src2, except;
    mips_decode decoder(alu_op, writeenable, rd_src, alu_src2, except,
                        opcode, funct);
endmodule // decoder_test

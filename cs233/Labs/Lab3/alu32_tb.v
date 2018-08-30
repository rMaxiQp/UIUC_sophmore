module alu32_test;
    reg [31:0] A = 0, B = 0;
    reg [2:0] control = 0;

    initial begin
        $dumpfile("alu32.vcd");
        $dumpvars(0, alu32_test);

             A = 2147483647; B = 893829; control = `ALU_ADD; // try adding 8 and 4
        # 10 A = 32'hffffffff; B = 1; control = `ALU_ADD;
        # 10 A = 2; B = 5; control = `ALU_SUB; // try subtracting 5 from 2
        # 10 A = 0; B = 0; control = `ALU_AND;
        # 10 A = 1; B = 14; control = `ALU_OR;
        # 10 A = 10; B = 133; control = `ALU_NOR;
        # 10 A = 555; B = 1123; control = `ALU_XOR;

        # 10 $finish;
    end

    wire [31:0] out;
    wire overflow, zero, negative;
    alu32 a(out, overflow, zero, negative, A, B, control);

    initial begin
        $display(" A  B  over neg  zero   control  out");
        $monitor("%d %d %d %d %d %d %d(at time %t)", A, B, overflow, negative, zero, control, out, $time);
    end

endmodule // alu32_test

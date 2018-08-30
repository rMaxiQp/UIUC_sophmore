module i_reader_test;
    reg clk = 0;
    always #5 clk = !clk;

    reg [1:0] bits = 2'b00;
    reg       restart;

    initial begin
        $dumpfile("ir.vcd");
        $dumpvars(0, i_reader_test);

            restart = 1;
        # 23
            bits = 2'b11;
        # 4
            bits = 2'b00;       // not an I: restart is still 1
        # 5
            restart = 0;
        # 8
            bits = 2'b11;
        # 10
            bits = 2'b00;       // I
        # 10
            bits = 2'b11;
        # 20
            bits = 2'b00;       // not an I: sequence was 00, 11, 11, 00
        # 15
            $finish;
    end

    wire I;
    i_reader ir(I, bits, clk, restart);

    initial
        $monitor("Time %t: bits = %b I = %d restart = %d",
            $time, bits, I, restart);
endmodule // word_reader_test

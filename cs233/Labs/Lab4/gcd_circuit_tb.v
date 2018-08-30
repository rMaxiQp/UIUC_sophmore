module gcd_circuit_test;
    reg       clock = 0;
    always #1 clock = !clock;
    reg [31:0] X = 0, Y = 0;
    reg reset = 1;
    reg x_sel = 0, y_sel = 0, x_en = 0, y_en = 0, output_en = 0;

    initial begin
        $dumpfile("gcd_circuit.vcd");
        $dumpvars(0, gcd_circuit_test);
        #1      reset = 0;

        // Read X/Y
        #5     X = 354; Y = 118; x_en = 1; y_en = 1;
        // Stop reading X/Y
        #2      x_sel = 0; x_en = 0; y_sel = 0; y_en = 0;
        // Observe X
        #2      output_en = 1;
        #2      output_en = 0;
        // Do X = X - Y
        #2      x_en = 1; x_sel = 1;
        //#2      x_en = 0; x_sel = 0;
        // Observe X
        #2      output_en = 1;
        #2      output_en = 0;

        #2      x_en = 1; x_sel = 1;
        #2      x_en = 0; x_sel = 0;

        #2      output_en = 1;
        #2      output_en = 0;

        #2      x_en = 1; x_sel = 1;
        #2      x_en = 0; x_sel = 0;

        #2      output_en = 1;
        #2      output_en = 0;
        #4
/*
        //another test case
        #1      reset = 0;
        // Read X/Y
        #2     X = 9; Y = 6; x_en = 1; y_en = 1;
        // Stop reading X/Y
        #2      x_sel = 0; x_en = 0; y_sel = 0; y_en = 0;
        // Observe X
        #2      output_en = 1;
        #2      output_en = 0;
        // Do X = X - Y
        #2      x_en = 1; x_sel = 1;
        #2      x_en = 0; x_sel = 0;
        // Observe X
        #2      output_en = 1;
        #2      output_en = 0;
        #4
*/
        $finish;
    end

    wire [31:0] out;
    wire x_lt_y, x_ne_y;

    gcd_circuit circuit(out, x_lt_y, x_ne_y, X, Y, x_sel, y_sel, x_en, y_en, output_en, clock, reset);
    initial begin
      $display("out, x_lt_y, x_ne_y, X, Y, x_sel, y_sel, x_en, y_en, output_en, clock, reset");
      $monitor("%d %d %d %d %d %d %d %d %d %d %d %d (at time%t)", out, x_lt_y, x_ne_y, X, Y, x_sel, y_sel, x_en, y_en, output_en, clock, reset, $time);
    end


endmodule

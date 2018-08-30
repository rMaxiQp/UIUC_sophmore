module gcd_control_test;
    reg       clock = 0;
    always #1 clock = !clock;
    reg [31:0] X = 0, Y = 0;
    reg go = 0;
    reg reset = 1;

    initial begin
        $dumpfile("gcd_control.vcd");
        $dumpvars(0, gcd_control_test);
        #1      reset = 0;
        #5      X = 354; Y = 118; go = 1;
        // You may have to change the following # 100 if it isn't
        // enough time for your machine to finish processing.
        #100    go = 0;
        #2      reset = 1;
        #1      reset = 0;
	      #10     X=20;Y=31;go=1;

	      #100    go=0;
        //#2      reset = 1;
        //#1      reset = 0;
        #5      X = 15; Y =25; go =1;
        #100    go = 0;
        $finish;
    end

    wire [31:0] out;
	wire done, x_lt_y, x_ne_y, x_sel, y_sel, x_en, y_en, output_en;
    gcd_circuit circuit(out, x_lt_y, x_ne_y, X, Y, x_sel, y_sel, x_en, y_en, output_en, clock, reset);
    gcd_control control(done, x_sel, y_sel, x_en, y_en, output_en, go, x_lt_y, x_ne_y, clock, reset);

    initial begin
      $display("X Y Out");
      $monitor("%d %d %d (at time %t)", X, Y, out, $time);
    end
endmodule

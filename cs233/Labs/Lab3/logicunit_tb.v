module logicunit_test;
    reg A = 0;
    always #1 A = !A;
    reg B = 0;
    always #2 B = !B;

    reg [1:0] control = 0;

    initial begin
        $dumpfile("logicunit.vcd");
        $dumpvars(0, logicunit_test);

        // control is initially 0
        # 16 control = 1; // wait 16 time units (why 16?) and then set it to 1
        # 16 control = 2; // wait 16 time units and then set it to 2
        # 16 control = 3; // wait 16 time units and then set it to 3
        # 16 $finish; // wait 16 time units and then end the simulation
    end

    wire out;
    logicunit l4(out, A, B, control);

    /*
    initial begin
        $display("A B C D s o");
        $monitor("%d %d %d %d (at time %t)", A, B, control, out, $time);
    end
    */
endmodule // logicunit_test

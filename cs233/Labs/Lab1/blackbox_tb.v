module blackbox_test;

     reg a_in, b_in, c_in;
     wire h_out;
     blackbox bb1 (h_out, a_in, b_in, c_in);

     initial begin

          $dumpfile("blackbox.vcd");
          $dumpvars(0,blackbox_test);

          a_in = 0; b_in = 0; c_in = 0; # 10;
          a_in = 0; b_in = 0; c_in = 1; # 10;
          a_in = 0; b_in = 1; c_in = 0; # 10;
          a_in = 0; b_in = 1; c_in = 1; # 10;
          a_in = 1; b_in = 0; c_in = 0; # 10;
          a_in = 1; b_in = 0; c_in = 1; # 10;
          a_in = 1; b_in = 1; c_in = 0; # 10;
          a_in = 1; b_in = 1; c_in = 1; # 10;

               $finish;
     end

     initial
          $monitor("At time %2t, a_in = %d b_in = %d c_in = %d h_out = %d", $time, a_in, b_in, c_in, h_out);

endmodule // blackbox_test

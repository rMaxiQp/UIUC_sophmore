module test;

   // these are inputs to "circuit under test"
   reg [1:0] a;
   reg [1:0] b;
  // wires for the outputs of "circuit under test"
   wire [1:0] out;
  // the circuit under test
   udiv u(out, a, b);  
    
   initial begin               // initial = run at beginning of simulation
                               // begin/end = associate block with initial
      
      $dumpfile("test.vcd");  // name of dump file to create
      $dumpvars(0, test);     // record all signals of module "test" and sub-modules
                              // remember to change "test" to the correct
                              // module name when writing your own test benches
        
      // test all input combinations
      a = 0; b = 0; #10;
      a = 0; b = 1; #10;
      a = 0; b = 2; #10;
      a = 0; b = 3; #10;
      a = 1; b = 0; #10;
      a = 1; b = 1; #10;
      a = 1; b = 2; #10;
      a = 1; b = 3; #10;
      a = 2; b = 0; #10;
      a = 2; b = 1; #10;
      a = 2; b = 2; #10;
      a = 2; b = 3; #10;
      a = 3; b = 0; #10;
      a = 3; b = 1; #10;
      a = 3; b = 2; #10;
      a = 3; b = 3; #10;
      
      $finish;        // end the simulation
   end                      
   
   initial begin
     $display("inputs = a, b  outputs = out");
     $monitor("inputs = %b  %b  outputs = %b   time = %2t",
              a, b, out, $time);
   end
endmodule // test

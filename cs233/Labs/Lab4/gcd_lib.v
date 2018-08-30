module dffe(q, d, clk, enable, reset);
    output q;
    reg    q;
    input  d;
    input  clk, enable, reset;
    always@(reset)
      if (reset == 1'b1)
        q <= 0;
    always@(posedge clk)
      if ((reset == 1'b0) && (enable == 1'b1))
        q <= d;
endmodule // dffe

module register(q, d, clk, enable, reset);
    parameter
        width = 32,
        reset_value = 0;
 
    output [(width-1):0] q;
    reg    [(width-1):0] q;
    input  [(width-1):0] d;
    input                clk, enable, reset;
 
    always@(reset)
      if (reset == 1'b1)
        q <= reset_value;
 
    always@(posedge clk)
      if ((reset == 1'b0) && (enable == 1'b1))
        q <= d;

endmodule // register

module comparator(lt, ne, A, B);
  output    lt, ne;
  input     [31:0] A, B;
  assign lt = A < B;
  assign ne = A != B;
endmodule

module subtractor(out, A, B);
  output    [31:0] out;
  input     [31:0] A, B;
  assign out = A-B;
endmodule
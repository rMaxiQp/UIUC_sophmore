module keypad(valid, number, a, b, c, d, e, f, g);
   output valid;
   output [3:0] number;
   input  a, b, c, d, e, f, g;
   wire w1, w2, w3, w4, w5, w6, w7, w8, w9, w0;

   or o12(number[3], w8, w9);
   or o23(number[2], w4, w5, w6, w7);
   or o34(number[1], w2, w3, w6, w7);
   or o45(number[0], w1, w3, w5, w7, w9);

   or o1(valid, w1, w2, w3, w4, w5, w6, w7, w8, w9, w0);
   and a11(w1, a, d);
   and a12(w4, a, e);
   and a13(w7, a, f);
   and a21(w2, b, d);
   and a22(w5, b, e);
   and a23(w8, b, f);
   and a24(w0, b, g);
   and a31(w3, c, d);
   and a32(w6, c, e);
   and a33(w9, c, f);

endmodule // keypad

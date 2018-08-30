// Design a circuit that divides a two 2-bit unsigned binary number
// ('a') by another 2-bit unsigned binary number ('b') to produce a
// 2-bit output ('out').  When dividing by 0, set the output to all
// ones.

module udiv(out, a, b);
   output [1:0] out;
   input  [1:0]	a, b;
   wire w_zero, w1, w2, w3, w4;

   and a1(w_zero,!b[0], !b[1]);
   and a2(w1, a[0], a[1]);
   and a3(w2, w1, !b[1]);
   and a4(w3, a[1], !a[0], b[0], !b[1]);
   and a5(w4, !a[0]^b[0], !a[1]^b[1]);

   or o88(out[0], w_zero, w1, w2, w4);
   or o99(out[1], w_zero, w2, w3);
endmodule // udiv

// GCD datapath
module gcd_circuit(out, x_lt_y, x_ne_y, X, Y, x_sel, y_sel, x_en, y_en, output_en, clock, reset);
	output  [31:0] out;
	output  x_lt_y, x_ne_y;
	input	[31:0]	X, Y;
	input   x_sel, y_sel, x_en, y_en, output_en, clock, reset;
	wire [31:0] w_tmpX, w_tmpY, w1, w2, w_x, w_y, w_temp;

	mux2v m1(w_tmpX, X, w1, x_sel);
	mux2v m2(w_tmpY, Y, w2, y_sel);
	register r1(w_x, w_tmpX, clock, x_en, reset);
	register r2(w_y, w_tmpY, clock, y_en, reset);
	subtractor s1(w1, w_x, w_y);
	subtractor s2(w2, w_y, w_x);
	comparator c1(x_lt_y, x_ne_y, w_x, w_y);
	register r3(out, w_x, clock, output_en, reset);

endmodule // gcd_circuit

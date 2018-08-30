module gcd_control(done, x_sel, y_sel, x_en, y_en, output_en, go, x_lt_y, x_ne_y, clock, reset);
	output	x_sel, y_sel, x_en, y_en, output_en, done;
	input	go, x_lt_y, x_ne_y;
	input	clock, reset;
	wire w_garbage, w_input, w_g1_x, w_g0_x,
			 w_g1_y, w_g0_y, w_g1_d, w_g0_d, w_g0_e, w_g1_e, w_g0_c, w_g1_c,
			 w_garbage_next, w_input_next, w_g1_x_next, w_g0_c_next, w_g1_c_next,
			 w_g0_x_next, w_g1_y_next, w_g0_y_next, w_g1_d_next, w_g0_d_next, w_g0_e_next, w_g1_e_next;

	 assign w_garbage_next = (reset)|(~go & w_garbage) | (go & (w_g0_x | w_g0_c | w_g0_y | w_g0_e));
	 assign w_input_next = (w_garbage | w_g0_d) & go & ~reset;

	 assign w_g0_e_next = (w_g1_x | w_g1_y |w_input | w_g0_x | w_g0_y) & ~go & ~reset;
	 assign w_g1_e_next = go & (w_g1_x | w_g1_y | w_input) & ~reset;

	 assign w_g0_c_next = ((w_g0_e & x_ne_y) | w_g1_e) & ~go & ~reset;
	 assign w_g1_c_next = w_g1_e & go & x_ne_y & ~reset;

	 assign w_g0_x_next = ((~x_lt_y & w_g0_c) | w_g1_x) & ~go & ~reset;
	 assign w_g1_x_next = w_g1_c & go & ~x_lt_y & ~reset;

	 assign w_g0_y_next = ((x_lt_y & w_g0_c) | w_g1_y) & ~go & ~reset;
	 assign w_g1_y_next = w_g1_c & go & x_lt_y & ~reset;

	 assign w_g0_d_next = (w_g0_d | ((w_g1_e | w_g0_e) & ~x_ne_y)) & ~go & ~reset;
	 assign w_g1_d_next = ~x_ne_y & go & (w_g1_e| w_g1_d) & ~reset;

	 dffe fsGarbage(w_garbage, w_garbage_next, clock, 1'b1, 1'b0);
	 dffe fsInput(w_input, w_input_next, clock, 1'b1, 1'b0);
	 dffe fsG1X(w_g1_x, w_g1_x_next, clock, 1'b1, 1'b0);
	 dffe fsG0X(w_g0_x, w_g0_x_next, clock, 1'b1, 1'b0);
	 dffe fsG1Y(w_g1_y, w_g1_y_next, clock, 1'b1, 1'b0);
	 dffe fsG0Y(w_g0_y, w_g0_y_next, clock, 1'b1, 1'b0);
	 dffe fsG1D(w_g1_d, w_g1_d_next, clock, 1'b1, 1'b0);
	 dffe fsG0D(w_g0_d, w_g0_d_next, clock, 1'b1, 1'b0);
	 dffe fsG1E(w_g1_e, w_g1_e_next, clock, 1'b1, 1'b0);
	 dffe fsG0E(w_g0_e, w_g0_e_next, clock, 1'b1, 1'b0);
	 dffe fsG1C(w_g1_c, w_g1_c_next, clock, 1'b1, 1'b0);
	 dffe fsG0C(w_g0_c, w_g0_c_next, clock, 1'b1, 1'b0);

	 assign x_sel = ~w_input;
	 assign y_sel = ~w_input;
	 assign x_en = ~done & (w_input | w_g1_x | w_g0_x);
	 assign y_en = ~done & (w_input | w_g1_y | w_g0_y);
	 assign output_en = ~done;
	 assign done = w_g1_d | w_g0_d;
endmodule //GCD_control

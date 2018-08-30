// dffe: D-type flip-flop with enable
//
// q      (output) - Current value of flip flop
// d      (input)  - Next value of flip flop
// clk    (input)  - Clock (positive edge-sensitive)
// enable (input)  - Load new value? (yes = 1, no = 0)
// reset  (input)  - Asynchronous reset   (reset =  1)
//
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


module i_reader(I, bits, clk, restart);
    output      I;
    input [1:0] bits;
    input       restart, clk;
    wire        in00, in11,
                sGarbage, sGarbage_next,
                sBlank, sBlank_next,
                sI, sI_next,
                sI_end, sI_end_next;

    assign in00 = bits == 2'b00;
    assign in11 = bits == 2'b11;

    assign sGarbage_next = ((sBlank | sI_end) & ~(in00 | in11)) | ((sI | sGarbage) & ~in00);
    assign sBlank_next = (sBlank | sI_end | sGarbage) & in00;
    assign sI_next = (sBlank | sI_end) & in11;
    assign sI_end_next = sI & in00;

    // enable hardcoded to 1, reset hardcoded to 0
    dffe fsGarbage(sGarbage, sGarbage_next, clk, 1'b1, 1'b0);
    dffe fsBlank(sBlank, sBlank_next, clk, 1'b1, 1'b0);
    dffe fsI(sI, sI_next, clk, 1'b1, 1'b0);
    dffe fsI_end(sI_end, sI_end_next, clk, 1'b1, 1'b0);

    // outputs are associated with states, NOT transitions
    assign I = sI_end;
endmodule // word_reader

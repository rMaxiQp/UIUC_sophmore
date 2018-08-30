`define STATUS_REGISTER 5'd12
`define CAUSE_REGISTER  5'd13
`define EPC_REGISTER    5'd14

module cp0(rd_data, EPC, TakenInterrupt,
           regnum, wr_data, next_pc, TimerInterrupt,
           MTC0, ERET, clock, reset);
    output [31:0] rd_data;
    output [29:0] EPC;
    output        TakenInterrupt;
    input   [4:0] regnum;
    input  [31:0] wr_data;
    input  [29:0] next_pc;
    input         TimerInterrupt, MTC0, ERET, clock, reset;

    wire [29:0] epc_D;
    wire [31:0] user_status, cause_register, status_register, dc_out;
    wire exp_lvl, epc_enable, us_enable, el_reset;

    assign epc_enable = TakenInterrupt | dc_out[14];
    assign el_reset = reset | ERET;
    assign TakenInterrupt = (cause_register[15] & status_register[15]) & (~status_register[1] & status_register[0]);
    assign status_register = {16'b0, user_status[15:8], 6'b0, exp_lvl, user_status[0]};
    assign cause_register = {16'b0, TimerInterrupt, 15'b0};

    assign rd_data = {{32{regnum == `STATUS_REGISTER }} & status_register} |
    {{32{regnum == `CAUSE_REGISTER}} & cause_register} | {{32{regnum == `EPC_REGISTER}} & {EPC, 2'b0}};

    decoder32 dc(dc_out, regnum, MTC0);

    mux2v #(30) m1(epc_D, wr_data[31:2], next_pc, TakenInterrupt);

    register us(user_status, wr_data, clock, dc_out[12], reset);
    register #(30) epc(EPC, epc_D, clock, epc_enable, reset);

    dffe el(exp_lvl, 1'b1, clock, TakenInterrupt, el_reset);

endmodule

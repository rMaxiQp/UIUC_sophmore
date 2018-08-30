module timer(TimerInterrupt, TimerAddress, cycle,
             address, data, MemRead, MemWrite, clock, reset);
    output        TimerInterrupt, TimerAddress;
    output [31:0] cycle;
    input  [31:0] address, data;
    input         MemRead, MemWrite, clock, reset;

    wire [31:0] Q_IC, Q_CC, D_CC;
    wire timerWrite, timerRead, acknowledge, il_reset;
    wire addrCheck, ackCheck, il_enable;

    assign addrCheck = address == 32'hffff001c;
    assign ackCheck = address == 32'hffff006c;
    assign il_enable = Q_IC == Q_CC;

    or ir(il_reset, reset, acknowledge);
    or ta(TimerAddress, addrCheck, ackCheck);
    and tr(timerRead, MemRead, addrCheck);
    and tw(timerWrite, MemWrite, addrCheck);
    and ack(acknowledge, MemWrite, ackCheck);

    tristate tc(cycle, Q_CC, timerRead);

    dffe il(TimerInterrupt, 1'b1, clock, il_enable ,il_reset);

    register #(32,32'hffffffff) ic(Q_IC, data, clock, timerWrite, reset);
    register cc(Q_CC, D_CC, clock, 1'b1, reset);

    alu32 plus(D_CC, , ,`ALU_ADD, Q_CC, 32'b1);
    // HINT: make your interrupt cycle register reset to 32'hffffffff
    //       (using the reset_value parameter)
    //       to prevent an interrupt being raised the very first cycle
endmodule

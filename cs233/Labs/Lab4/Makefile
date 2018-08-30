.PHONY: all clean gcd_control rf
all:
	@echo Please use one of the make targets specified in the handout.

gcd_circuit: gcd_circuit_exe
	./$<

gcd_circuit_exe: gcd_circuit_tb.v gcd_circuit.v mux_lib.v gcd_lib.v
	iverilog -o $@ -Wall $^


gcd_control: gcd_control_exe
	./$<

gcd_control_exe: gcd_control_tb.v gcd_control.v gcd_circuit.v mux_lib.v gcd_lib.v
	iverilog -o $@ -Wall $^


rf: rf_exe
	./$<

rf_exe: rf.v rf_tb.v mux_lib.v
	iverilog -o $@ -Wall $^


clean:
	rm -f *_exe *.vcd

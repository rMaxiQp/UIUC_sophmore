.text

## unsigned int
## circular_shift(unsigned int in, unsigned char s) {
##     return (in >> 8 * s) | (in << (32 - 8 * s));
## }

.globl circular_shift
circular_shift:


	mul $t1, $a1, 8
	srl $t0, $a0, $t1

	li  $t2, 32
	sub $t1, $t2, $t1
	sll $t2, $a0, $t1

	or  $v0, $t0, $t2
	jr	$ra

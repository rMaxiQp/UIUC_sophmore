.data

# Strings for printing purposes
t1_msg: .asciiz "key_addition(in1, in2, out): "
in1: .byte 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15
in2: .byte 12 14 1 3 5 7 9 11 13 15 0 2 4 6 8 10
out: .space 16

.text
MAIN_STK_SPC = 8
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)
	sw	$s0, 4($sp)

	la	$a0, t1_msg
	jal	print_string
	la	$a0, in1
	la	$a1, in2
	la	$a2, out
	jal	key_addition
	li	$s0, 0
main_for:
	bge	$s0, 16, main_cont
	la	$t0, out
	add	$t0, $t0, $s0
	lbu	$a0, 0($t0)
	jal	print_int_and_space
	add	$s0, $s0, 1
	j	main_for
main_cont:
	jal	print_newline

	lw	$ra, 0($sp)
	lw	$s0, 4($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

.data

# Strings for printing purposes
t1_msg: .asciiz "circular_shift(0, 128): "
t2_msg: .asciiz "circular_shift(2095, 6): "
t3_msg: .asciiz "circular_shift(30, 0): "

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, t1_msg
	jal	print_string
	li	$a0, 0
	li	$a1, 128
	jal	circular_shift
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, t2_msg
	jal	print_string
	li	$a0, 2095
	li	$a1, 6
	jal	circular_shift
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, t3_msg
	jal	print_string
	li	$a0, 30
	li	$a1, 0
	jal	circular_shift
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

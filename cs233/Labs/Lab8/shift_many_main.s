.data
null_str:	.asciiz "NULL "

.align 2
test1:		.word 0xDEADBEEF test1_1 test1_2 test1_3 test1_4
test1_1:	.word 0
test1_2:	.word 0
test1_3:	.word 0
test1_4:	.word 0
test1_str:	.asciiz "shift_many(s1, 0): "

.align 2
test2:		.word 0x00FF00AA test2_1 test2_2 test2_3 test2_4
test2_1:	.word 0
test2_2:	.word 0
test2_3:	.word 0
test2_4:	.word 0
test2_str:	.asciiz "shift_many(s2, 3): "

.align 2
test3:		.word 0x12345678 test3_1 0 test3_3 test3_4
test3_1:	.word 0
test3_3:	.word 0
test3_4:	.word 0
test3_str:	.asciiz "shift_many(s3, 2): "

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, test1
	li	$a1, 0
	jal	shift_many
	la	$a0, test1_str
	jal	print_string
	la	$a0, test1
	jal	print_shifter
	jal	print_newline

	la	$a0, test2
	li	$a1, 3
	jal	shift_many
	la	$a0, test2_str
	jal	print_string
	la	$a0, test2
	jal	print_shifter
	jal	print_newline

	la	$a0, test3
	li	$a1, 2
	jal	shift_many
	la	$a0, test3_str
	jal	print_string
	la	$a0, test3
	jal	print_shifter
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra


print_shifter:
	sub	$sp, $sp, 8
	sw	$ra, 0($sp)

	add	$a0, $a0, 4
	sw	$a0, 4($sp)

	lw	$t0, 0($a0)
	beq	$t0, $0, ps_1n
	lw	$a0, 0($t0)
	jal	print_int_hex_and_space
	j	ps_2

ps_1n:
	la	$a0, null_str
	jal	print_string

ps_2:
	lw	$a0, 4($sp)
	lw	$t0, 4($a0)
	beq	$t0, $0, ps_2n
	lw	$a0, 0($t0)
	jal	print_int_hex_and_space
	j	ps_3

ps_2n:
	la	$a0, null_str
	jal	print_string

ps_3:
	lw	$a0, 4($sp)
	lw	$t0, 8($a0)
	beq	$t0, $0, ps_3n
	lw	$a0, 0($t0)
	jal	print_int_hex_and_space
	j	ps_4

ps_3n:
	la	$a0, null_str
	jal	print_string

ps_4:
	lw	$a0, 4($sp)
	lw	$t0, 12($a0)
	beq	$t0, $0, ps_4n
	lw	$a0, 0($t0)
	jal	print_int_hex_and_space
	j	ps_done

ps_4n:
	la	$a0, null_str
	jal	print_string


ps_done:
	lw	$ra, 0($sp)
	add	$sp, $sp, 8
	jr	$ra

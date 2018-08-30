.data

# Strings for printing purposes
str1_msg1: .asciiz "nth_uniq_char(str1, 4): "
str1_msg2: .asciiz "nth_uniq_char(str1, 12): "
str2_msg1: .asciiz "nth_uniq_char(str2, 2): "
str2_msg2: .asciiz "nth_uniq_char(str2, 6): "

# Arrays
str1: .asciiz "abcdefghijk"
str2: .asciiz "aaabbbcccde"

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, str1_msg1
	jal	print_string
	la	$a0, str1
	li	$a1, 4
	jal	nth_uniq_char
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, str1_msg2
	jal	print_string
	la	$a0, str1
	li	$a1, 12
	jal	nth_uniq_char
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, str2_msg1
	jal	print_string
	la	$a0, str2
	li	$a1, 2
	jal	nth_uniq_char
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, str2_msg2
	jal	print_string
	la	$a0, str2
	li	$a1, 6
	jal	nth_uniq_char
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

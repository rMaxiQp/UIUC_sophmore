.data

# Strings for printing purposes
str1_msg1: .asciiz "max_unique_n_substr(str1, result, 2): "
str1_msg2: .asciiz "max_unique_n_substr(str1, result, 3): "
str1_msg3: .asciiz "max_unique_n_substr(str1, result, 4): "
str2_msg1: .asciiz "max_unique_n_substr(str2, result, 3): "

# Arrays
str1: .asciiz "xxxyziiijzz"
str2: .asciiz "xxxyziiijzzy"

dst1: .space 32
dst2: .space 32
dst3: .space 32
dst4: .space 32

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, str1_msg1
	jal	print_string
	la	$a0, str1
	la	$a1, dst1
	li	$a2, 2
	jal	max_unique_n_substr
	la	$a0, dst1
	jal	print_string
	jal	print_newline

	la	$a0, str1_msg2
	jal	print_string
	la	$a0, str1
	la	$a1, dst2
	li	$a2, 3
	jal	max_unique_n_substr
	la	$a0, dst2
	jal	print_string
	jal	print_newline

	la	$a0, str1_msg3
	jal	print_string
	la	$a0, str1
	la	$a1, dst3
	li	$a2, 4
	jal	max_unique_n_substr
	la	$a0, dst3
	jal	print_string
	jal	print_newline

	la	$a0, str2_msg1
	jal	print_string
	la	$a0, str2
	la	$a1, dst4
	li	$a2, 3
	jal	max_unique_n_substr
	la	$a0, dst4
	jal	print_string
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

.data

# Strings for printing purposes
str_msg: .asciiz "my_strlen(str): "
str1_msg: .asciiz "my_strlen(str1): "

# Arrays
str: .asciiz "Hello world"
str1: .asciiz ""

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, str_msg
	jal	print_string
	la	$a0, str
	jal	my_strlen
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, str1_msg
	jal	print_string
	la	$a0, str1
	jal	my_strlen
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

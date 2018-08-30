.data

# Strings for printing purposes
has_cycle_root1_str:
.asciiz "has_cycle(root1, 4): "
has_cycle_root2_str:
.asciiz "has_cycle(root2, 4): "
has_cycle_null_str:
.asciiz "has_cycle(NULL, 42): "

root1:          .word 0 root1_children
root1_children: .word a1 b1 0

a1:             .word 1 a1_children
a1_children:    .word 0

b1:             .word 2 b1_children
b1_children:    .word c1 0

c1:             .word 3 c1_children
c1_children:     .word 0

root2:          .word 0 root2_children
root2_children: .word a2 b2 0

a2:             .word 1 a2_children
a2_children:    .word c2 0

b2:             .word 2 b2_children
b2_children:    .word c2 0

c2:             .word 3 c2_children
c2_children:     .word 0


.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, has_cycle_root1_str
	jal	print_string
	la      $a0, root1
	li      $a1, 4
	jal	has_cycle
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

        la	$a0, has_cycle_root2_str
	jal	print_string
	la      $a0, root2
	li      $a1, 4
	jal	has_cycle
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline
	
	la	$a0, has_cycle_null_str
	jal	print_string
	move    $a0, $0
	li      $a1, 42
	jal	has_cycle
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline
	
	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

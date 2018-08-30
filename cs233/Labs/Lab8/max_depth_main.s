.data
# Strings for printing purposes
max_depth_linked_list_depth_5:
.asciiz "max_depth(tree1): "
max_depth_many_children_depth_2:
.asciiz "max_depth(tree2): "
max_depth_unbalanced_depth_4:
.asciiz "max_depth(tree3): "

tree1:    .word 0 arr_t1n1
arr_t1n1: .word t1n2 0
t1n2:     .word 1 arr_t1n2
arr_t1n2: .word t1n3 0
t1n3:     .word 2 arr_t1n3
arr_t1n3: .word t1n4 0
t1n4:     .word 3 arr_t1n4
arr_t1n4: .word t1n5 0
t1n5:     .word 4 arr_t1n5
arr_t1n5: .word 0

tree2:    .word 0 arr_t2n1
arr_t2n1: .word t2n2 t2n3 t2n4 t2n5 t2n6 0
t2n2:     .word 1 arr_t2n2
arr_t2n2: .word 0
t2n3:     .word 2 arr_t2n3
arr_t2n3: .word 0
t2n4:     .word 3 arr_t2n4
arr_t2n4: .word 0
t2n5:     .word 4 arr_t2n5
arr_t2n5: .word 0
t2n6:     .word 5 arr_t2n6
arr_t2n6: .word 0

tree3:    .word 0 arr_t3n1
arr_t3n1: .word t3n2 t3n3 t3n4 0
t3n2:     .word 1 arr_t3n2
arr_t3n2: .word 0
t3n3:     .word 2 arr_t3n3
arr_t3n3: .word t3n5 0
t3n4:     .word 3 arr_t3n4
arr_t3n4: .word t3n6 t3n7 0
t3n5:     .word 4 arr_t3n5
arr_t3n5: .word t3n8 t3n9 0
t3n6:     .word 5 arr_t3n6
arr_t3n6: .word 0
t3n7:     .word 6 arr_t3n7
arr_t3n7: .word 0
t3n8:     .word 7 arr_t3n8
arr_t3n8: .word 0
t3n9:     .word 8 arr_t3n9
arr_t3n9: .word 0

.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	la	$a0, max_depth_linked_list_depth_5
	jal	print_string
	la	$a0, tree1
	jal	max_depth
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, max_depth_many_children_depth_2
	jal	print_string
	la	$a0, tree2
	jal	max_depth
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	la	$a0, max_depth_unbalanced_depth_4
	jal	print_string
	la	$a0, tree3
	jal	max_depth
	move	$a0, $v0
	jal	print_int_and_space
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra

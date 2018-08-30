.data

.text
## struct Node {
##     int node_id;            // Unique node ID
##     struct Node **children; // pointer to null terminated array of children node pointers
## };
##
## int
## has_cycle(Node *root, int num_nodes) {
##     if (!root)
##         return 0;
##
##     Node *stack[num_nodes];
##     stack[0] = root;
##     int stack_size = 1;
##
##     int discovered[num_nodes];
##     for (int i = 0; i < num_nodes; i++) {
##         discovered[i] = 0;
##     }
##
##     while (stack_size > 0) {
##         Node *node_ptr = stack[--stack_size];
##
##         if (discovered[node_ptr->node_id]) {
##             return 1;
##         }
##         discovered[node_ptr->node_id] = 1;
##
##         for (Node **edge_ptr = node_ptr->children; *edge_ptr; edge_ptr++) {
##             stack[stack_size++] = *edge_ptr;
##         }
##     }
##
##     return 0;
## }

.globl has_cycle
has_cycle:
	move$v0, $0
	beq $a0, $0, end

	sub $sp, $sp, 12
	sw  $ra, 0($sp)
	sw  $s0, 4($sp)
	sw  $s1, 8($sp)

	mul $t9, $a1, 4
	sub $sp, $sp, $t9
	move$s0, $sp         #$s0 for stack
	sub $sp, $sp, $t9
	move$s1, $sp         #$s1 for discovered

	sw  $a0, 0($s0)      #stack = root
	li  $t1, 1           #$t1 for stack_size
	li  $t3, 0
	li  $v0, 1

for_loop:
	bge $t3, $a1, while_loop
	mul $t4, $t3, 4
	add $t4, $t4, $s1
	sw  $0, 0($t4)
	add $t3, $t3, 1
	j for_loop

while_loop:
	ble $t1, $0, change #$t1 for stack_size
	sub $t1, $t1, 1
	mul $t3, $t1, 4
	add $t4, $t3, $s0
	lw  $t4, 0($t4)     #Node* node_ptr = stack[--stack_size]
	lw  $t3, 0($t4)     #$t3 = node_ptr->node_id
	mul $t3, $t3, 4
	add $t3, $t3, $s1   #discovered[node_ptr->node_id]
	lw  $t5, 0($t3)
	bne $t5, $0, set    #return 1
	li  $t5, 1
	sw  $t5, 0($t3)     #discovered[node_ptr->node_id] = 1
	lw  $t7, 4($t4)     #Node **edge_ptr = node_ptr->children

for:
	lw  $t4, 0($t7)     #$t4 for edge_ptr
	beq $t4, $0, while_loop
	mul $t5, $t1, 4
	add $t5, $t5, $s0   #stack[stack_size]
	sw  $t4, 0($t5)
	add $t1, $t1, 1     #stack_size++
	add $t7, $t7, 4     #edge_ptr++
	j for

change:
	move $v0, $0

set:
	add $sp, $sp, $t9
	add $sp, $sp, $t9
	lw  $ra, 0($sp)
	lw  $s0, 4($sp)
	lw  $s1, 8($sp)
	add $sp, $sp, 12

end:
	jr	$ra

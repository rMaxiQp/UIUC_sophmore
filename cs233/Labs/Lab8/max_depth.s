.data

.text

## struct Node {
##     int node_id;            // Unique node ID
##     struct Node **children; // pointer to null terminated array of children node pointers
## };
##
## int
## max_depth(Node *current) {
##     if (current == NULL)
##         return 0;
##
##     int cur_child = 0;
##     Node *child = current->children[cur_child];
##     int max = 0;
##     while (child != NULL) {
##         int depth = max_depth(child);
##         if (depth > max)
##             max = depth;
##         cur_child++;
##         child = current->children[cur_child];
##     }
##     return 1 + max;
## }

.globl max_depth
max_depth:
	move $v0, $0
	beq $a0, $0, end

	sub $sp, $sp, 16
	sw  $ra, 0($sp)
	sw  $s0, 4($sp)     #$s0 for $a0
	sw  $s1, 8($sp)     #$s1 for cur_child
	sw  $s2, 12($sp)    #$s2 for max
	move$s0, $a0

	move$s1, $0         #$s1 = cur_child
	lw  $t1, 4($a0)
	lw  $t2, 0($t1)     #$t2 for child
	move$s2, $0

while:
	beq $t2, $0, set
	move$a0, $t2
	jal		max_depth			#jump to max_depth
	move$t0, $v0
	bgt $t0, $s2, if
	j loop

loop:
	add $s1, $s1, 1
	mul $t3, $s1, 4
	lw  $t4, 4($s0)
	add $t1, $t4, $t3
	lw  $t2, 0($t1)
	j while

if:
	move$s2, $t0
	j loop

set:
	add $v0, $s2, 1
	move$a0, $s0
	lw  $ra, 0($sp)
	lw  $s0, 4($sp)
	lw  $s1, 8($sp)
	lw  $s2, 12($sp)
	add $sp, $sp, 16

end:
	jr	$ra

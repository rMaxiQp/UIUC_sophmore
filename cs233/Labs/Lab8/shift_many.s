## struct Shifter {
##     unsigned int value;
##     unsigned int *to_rotate[4];
## };
##
##
## void
## shift_many(Shifter *s, int offset) {
##     for (int i = 0; i < 4; i++) {
##         unsigned int *ptr = s->to_rotate[i];
##
##         if (ptr == NULL) {
##             continue;
##         }
##
##         unsigned char x = (i + offset) & 3;
##         *ptr = circular_shift(s->value, x);
##     }
## }

shift_many:
	sub $sp, $sp, 24
	sw  $ra, 0($sp)
	sw  $s0, 4($sp)    #s0 for $a0
	sw  $s1, 8($sp)    #s1 for #a1
	sw  $s2, 12($sp)
	sw  $s3, 16($sp)
	sw  $s4, 20($sp)
	move$s0, $a0
	move$s1, $a1

	li  $s2, 0         #$s2 for i

for:
	bge $s2, 4, end
	move$t0, $s2
	mul $t2, $s2, 4
	add $t3, $s0, $t2  #s->to_rotate[i]
	add $t3, $t3, 4
	lw  $s3, 0($t3)    #$s3 for ptr
	add $s2, $s2, 1

	beq $s3, $0, for
	add $s4, $t0, $s1  #i + offset
	and $s4, $s4, 3    #$t4 = x
	lw  $a0, 0($s0)    #s->value
	move$a1, $s4
	jal circular_shift
	sw  $v0, 0($s3)
	j for

end:
	move$a0, $s0
	move$a1, $s1
	lw  $ra, 0($sp)
	lw  $s0, 4($sp)
	lw  $s1, 8($sp)
	lw  $s2, 12($sp)
	lw  $s3, 16($sp)
	lw  $s4, 20($sp)
	add $sp, $sp, 24
	jr  $ra

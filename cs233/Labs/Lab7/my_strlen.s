.text

## unsigned int
## my_strlen(char *in) {
##     if (!in)
##         return 0;
##
##     unsigned int count = 0;
##     while (*in) {
##         count++;
##         in++;
##     }
##
##     return count;
## }

.globl my_strlen
my_strlen:
	move$v0, $0
	bne $a0, $0, ini_
	jr  $ra

ini_:
	move$t0, $a0

loop:
	lb  $t1, 0($a0)
	beq $t1, $0, end
	addi $v0, $v0, 1
	addi $a0, $a0, 1
	j loop

end:
	move $a0, $t0
	jr $ra

.data
uniq_chars: .space 256

.text

## int
## nth_uniq_char(char *in_str, int n) {
##     if (!in_str || !n)
##         return -1;
##
##     uniq_chars[0] = *in_str;
##     int uniq_so_far = 1;
##     int position = 0;
##     in_str++;
##     while (uniq_so_far < n && *in_str) {
##         char is_uniq = 1;
##         for (int j = 0; j < uniq_so_far; j++) {
##             if (uniq_chars[j] == *in_str) {
##                 is_uniq = 0;
##                 break;
##             }
##         }
##         if (is_uniq) {
##             uniq_chars[uniq_so_far] = *in_str;
##             uniq_so_far++;
##         }
##         position++;
##         in_str++;
##     }
##
##     if (uniq_so_far < n) {
##         position++;
##     }
##     return position;
## }

.globl nth_uniq_char
nth_uniq_char:
	li 	$t0, 1
	sub $v0, $0, $t0
	beq $a0, $0, end
	beq $a1, $0, end
	sub $sp, $sp, 4
	sw  $s0, 0($sp)
	move$s0, $a0

start:
	la $t0, uniq_chars
	lb $t1, 0($a0)          #*in_str
	sb $t1, 0($t0)          #$t0 for uniq_char[]
	li $t1, 1               #$t1 for uniq_so_far
	li $t2, 0               #$t2 for position
	add $a0, $a0, 1         #in_str++

while:
	lb  $t3, 0($a0)         #*in_str
	bge $t1, $a1, check     #uniq_so_far >= n
	beq $t3, $0,  check     #*in_str != NULL
	li  $t3, 1              #is_uniq = 1
	li	$t4, 0              #$t4 for j

for:
	add $t5, $t0, $t4				#$t5 for uniq_char[j]
	lb	$t6, 0($t5)					#$t6 for current value of uniq_char[j]
	lb  $t7, 0($a0)					#*in_str
	add $t4, $t4, 1					#j++
	beq $t6, $t7, bk	      #is_uniq == 0
	blt $t4, $t1, for
	j not_uniq

bk:
	li  $t3, 0
	beq $t3, $0, while_con

not_uniq:
	add $t4, $t1, $t0       #uniq_char[uniq_so_far]
	lb  $t3, 0($a0)         #*in_str
	sb  $t3, 0($t4)         #uniq_char[uniq_so_far] = *in_str
	add $t1, $t1, 1         #uniq_so_far++

while_con:
	add $t2, $t2, 1         #position++
	add $a0, $a0, 1         #in_str++
	j while

check:
	bge $t1, $a1, mv
	add $t2, $t2, 1

mv:
	move$v0, $t2
	move$a0, $s0
	lw  $s0, 0($sp)
	add $sp, $sp, 4

end:
	jr	$ra

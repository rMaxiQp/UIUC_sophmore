.text

## void
## max_unique_n_substr(char *in_str, char *out_str, int n) {
##     if (!in_str || !out_str || !n)
##         return;
##
##     char *max_marker = in_str;
##     unsigned int len_max = 0;
##     unsigned int len_in_str = my_strlen(in_str);
##     for (unsigned int cur_pos = 0; cur_pos < len_in_str; cur_pos++) {
##         char *i = in_str + cur_pos;
##         int len_cur = nth_uniq_char(i, n + 1);
##         if (len_cur > len_max) {
##             len_max = len_cur;
##             max_marker = i;
##         }
##     }
##
##     my_strncpy(out_str, max_marker, len_max);
## }

.globl max_unique_n_substr
max_unique_n_substr:

	beq $a0, $0, end  #$a0 = in_str
	beq $a1, $0, end  #$a1 = out_str
	beq $a2, $0, end  #$a2 = n

start:
	sub $sp, $sp, 32
	sw  $ra, 0($sp)   #save $ra
	sw  $s0, 4($sp)
	sw  $s1, 8($sp)
	sw  $s2, 12($sp)
	sw  $s3, 16($sp)
	sw  $s4, 20($sp)
	sw  $s5, 24($sp)
	sw  $s6, 28($sp)
	move$s0, $a0      #$s0 = $a0
	move$s1, $a1			#$s1 = $a1
	move$s2, $a2      #$s2 = $a2

	move$s3, $s0      #$s3 = max_marker
	lui $s4, 0        #$s4 = len_max
	jal my_strlen
	move$s5, $v0      #$s5 = len_in_str
	lui $s6, 0        #$s6 = cur_pos
	add $a2, $a2, 1

for:
	bge $s6, $s5, cpy
	move$a0, $s0
	add $a0, $a0, $s6 #$a0 = i
	move$a1, $a2
	jal nth_uniq_char
	add $s6, $s6, 1
	ble $v0, $s4, for #len_cur > len_max
	move$s4, $v0
	move$s3, $a0
	j for

cpy:
	move$a0, $s1      #out_str
	move$a1, $s3      #max_marker
	move$a2, $s4      #len_max
	jal my_strncpy
	lw  $ra, 0($sp)
	lw  $s0, 4($sp)
	lw  $s1, 8($sp)
	lw  $s2, 12($sp)
	lw  $s3, 16($sp)
	lw  $s4, 20($sp)
	lw  $s5, 24($sp)
	lw  $s6, 28($sp)
	add $sp, $sp, 32

end:
	jr	$ra

.data

.text

## void
## decrypt(unsigned char *ciphertext, unsigned char *plaintext, unsigned char *key,
##         unsigned char rounds) {
##     unsigned char A[16], B[16], C[16], D[16];
##     key_addition(ciphertext, &key[16 * rounds], C);
##     inv_shift_rows(C, (unsigned int *) B);
##     inv_byte_substitution(B, A);
##     for (unsigned int k = rounds - 1; k > 0; k--) {
##         key_addition(A, &key[16 * k], D);
##         inv_mix_column(D, C);
##         inv_shift_rows(C, (unsigned int *) B);
##         inv_byte_substitution(B, A);
##     }
##     key_addition(A, key, plaintext);
##     return;
## }

.globl decrypt
decrypt:
	sub $sp, $sp, 100
	sw  $s0, 0($sp)   #$s0
	sw  $s1, 4($sp)   #$s1
	sw  $s2, 8($sp)   #$s2
	sw  $s3, 12($sp)  #$s3
	sw  $s4, 16($sp)  #$s4
	sw  $s5, 20($sp)  #$s5
	sw  $s6, 24($sp)  #$s6
	sw  $s7, 28($sp)  #$s7
	sw  $ra, 32($sp)  #store $ra

	move$s1, $a1      #$s1 for $a1
	move$s2, $a2      #$s2 for $a2
	move$s3, $a3      #$s3 for $a3
	add $s4, $sp, 36  #$s4 for A
	add $s5, $sp, 52  #$s5 for B
	add $s6, $sp, 68  #$s6 for C
	add $s7, $sp, 84  #$s7 for D

	#key_addition
	mul $t0, $a3, 16
	add $a1, $t0, $a2
	move$a2, $s6
	jal key_addition

	#inv_shift_rows
	move$a0, $s6      #move C to $a0
	move$a1, $s5      #move B to $a1
	jal inv_shift_rows

	#inv_byte_substitution
	move$a0, $s5      #move B to $a0
	move$a1, $s4      #move A to $a1
	jal inv_byte_substitution

	subu $s0, $s3, 1   #$s0 for k

loop:
	ble $s0, $0, after
  #key_addition
	mul $t1, $s0, 16
	add $a1, $t1, $s2
	move$a0, $s4      #move A to $a0
	move$a2, $s7      #move D to $a2
	jal key_addition

	#inv_mix_column
	move$a0, $s7      #move D to $a0
	move$a1, $s6      #move C to $a1
	jal inv_mix_column

	#inv_shift_rows
	move$a0, $s6      #move C to $a0
	move$a1, $s5      #move B to $a1
	jal inv_shift_rows

	#inv_byte_substitution
	move$a0, $s5
	move$a1, $s4
	jal inv_byte_substitution

	sub $s0, $s0, 1
	j loop

after:
	move$a0, $s4      #move A to $a0
	move$a1, $s2      #move key to $a1
	move$a2, $s1      #move plaintext to $a2
	jal key_addition

	lw  $s0, 0($sp)   #$s0
	lw  $s1, 4($sp)   #$s1
	lw  $s2, 8($sp)   #$s2
	lw  $s3, 12($sp)  #$s3
	lw  $s4, 16($sp)  #$s4
	lw  $s5, 20($sp)  #$s5
	lw  $s6, 24($sp)  #$s6
	lw  $s7, 28($sp)  #$s7
	lw  $ra, 32($sp)  #load $ra
	add $sp, $sp, 100
	jr $ra

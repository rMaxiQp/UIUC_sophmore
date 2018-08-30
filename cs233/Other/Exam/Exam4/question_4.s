# before running your code for the first time, run:
#     module load QtSpim
# run with:
#     QtSpim -file main.s question_4.s

# void increment_vals(int **A, int *vals, int length) {
#     for (int i = 0; i < length; i += 2) {
#         vals[i] = *A[i] + 17;
#     }
# }
.globl increment_vals
increment_vals:
  li $t0, 0

loop:
  bge $t0, $a2, end
  mul $t1, $t0, 4
  add $t2, $t1, $a1 #vals[i]
  add $t3, $t1, $a0 #*A[i]
  lw  $t3, 0($t3)
  lw  $t3, 0($t3)
  add $t3, $t3, 17
  sw  $t3, 0($t2)
  add $t0, $t0, 2
  j loop

end:
  jr $ra

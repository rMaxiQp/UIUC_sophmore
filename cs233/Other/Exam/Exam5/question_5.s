# before running your code for the first time, run:
#     module load QtSpim
# run with:
#     QtSpim -file main.s question_5.s

# struct node_t {
#     node_t *left;
#     node_t *right;
#     int data;
# };
#
# void tweak_all(node_t *root, int zonk, int frood) {
#     if (root == NULL) {
#         return;
#     }
#
#     if (root->data != zonk) {
#         root->data = tweak(root->data, frood);
#     }
#
#     tweak_all(root->left, frood, zonk);
#     tweak_all(root->right, frood, zonk);
# }
.globl tweak_all
tweak_all:
  beq $a0, $0, return #if(root == NULL)

  sub $sp, $sp, 16
  sw  $ra, 0($sp)
  sw  $s0, 4($sp)
  sw  $s1, 8($sp)
  sw  $s2, 12($sp)

  move $s0, $a0
  move $s1, $a1
  move $s2, $a2

  lw  $t0, 8($a0)      #(root->data)
  beq $t0, $a1, recur
  move $a0, $t0
  move $a1, $s2
  jal tweak
  sw  $v0, 8($s0)

recur:
  lw $a0, 0($s0)
  move $a1, $s2
  move $a2, $s1
  jal tweak_all

  lw $a0, 4($s0)
  move $a1, $s2
  move $a2, $s1
  jal tweak_all

  move $a0, $s0
  move $a1, $s1
  move $a2, $s2
  lw $ra, 0($sp)
  lw $s0, 4($sp)
  lw $s1, 8($sp)
  lw $s2, 12($sp)
  add $sp, $sp, 16

return:
  jr $ra

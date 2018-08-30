.text

.globl my_strncpy
my_strncpy:
	sub	$sp, $sp, 16
	sw	$s0, 0($sp)
	sw	$s1, 4($sp)
	sw	$s2, 8($sp)
	sw	$ra, 12($sp)
	move	$s0, $a0
	move	$s1, $a1
	move	$s2, $a2

	move	$a0, $a1
	jal	my_strlen
	add	$v0, $v0, 1
	bge	$s2, $v0, my_strncpy_if
	move	$v0, $s2
my_strncpy_if:
	li	$t0, 0
my_strncpy_for:
	bge	$t0, $v0, my_strncpy_end
	add	$t1, $s1, $t0
	lb	$t2, 0($t1)
	add	$t1, $s0, $t0
	sb	$t2, 0($t1)
	add	$t0, $t0, 1
	j	my_strncpy_for
my_strncpy_end:
	lw	$s0, 0($sp)
	lw	$s1, 4($sp)
	lw	$s2, 8($sp)
	lw	$ra, 12($sp)
	add	$sp, $sp, 16
	jr	$ra

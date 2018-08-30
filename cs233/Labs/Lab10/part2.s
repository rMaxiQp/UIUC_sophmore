# syscall constants
PRINT_STRING	= 4
PRINT_CHAR	= 11
PRINT_INT	= 1

# memory-mapped I/O
VELOCITY	= 0xffff0010
ANGLE		= 0xffff0014
ANGLE_CONTROL	= 0xffff0018

BOT_X		= 0xffff0020
BOT_Y		= 0xffff0024

TIMER		= 0xffff001c

REQUEST_JETSTREAM	= 0xffff00dc
REQUEST_STARCOIN	= 0xffff00e0

PRINT_INT_ADDR		= 0xffff0080
PRINT_FLOAT_ADDR	= 0xffff0084
PRINT_HEX_ADDR		= 0xffff0088

# interrupt constants
BONK_MASK	= 0x1000
BONK_ACK	= 0xffff0060

TIMER_MASK	= 0x8000
TIMER_ACK	= 0xffff006c

REQUEST_STARCOIN_INT_MASK	= 0x4000
REQUEST_STARCOIN_ACK		= 0xffff00e4

.data
three:	.float	3.0
five:	.float	5.0
PI:	.float	3.141592
F180:	.float  180.0

.align 2
event_horizon_data: .space 90000
startcoin_data: .space 512

.text
main:
#enable interrupts
	li		$t4, REQUEST_STARCOIN_INT_MASK
	or		$t4, $t4, BONK_MASK
	or		$t4, $t4, 1
	mtc0	$t4, $12

	li		$t8, 1
	li  	$t5, 150

#request interrupt
	li		$t1, 10
	sw		$t1, VELOCITY

	la		$t7, event_horizon_data
	la 		$t0, startcoin_data
	sw		$t7, REQUEST_JETSTREAM
	sw 		$t0, REQUEST_STARCOIN

	li		$t3, 220
	li		$t4, 5
	li		$t5, 150
	li		$t6, 5
	li		$t7, 220

infinite:
	lw	$t9, startcoin_data
	lw  $t1, BOT_X
	lw  $t2, BOT_Y
	j angle_loop

angle_loop:
	bge $t1, 150, right
	bge $t2, 150, x_neg
	j		y_pos

right:
	bge $t2, 150, y_neg
	j		x_pos

x_pos:
	sub $a0, $t3, $t1  #delta x
	sub $a1, $t5, $t2  #delta y
	sub $t3, $t3, 3
	bgt $t3, 210, sb_arctan
	add $t3, $t3, 90
	j 	sb_arctan

x_neg:
	sub $a0, $t6, $t1  #delta x
	sub $a1, $t5, $t2  #delta y
	add $t6, $t6, 3
	blt $t6, 90, sb_arctan
	sub $t3, $t3, 80
	j 	sb_arctan

y_neg:
	sub $a0, $t5, $t1  #delta x
	sub $a1, $t7, $t2  #delta y
	sub $t7, $t7, 3
	bgt $t7, 210, sb_arctan
	add $t7, $t7, 90
	j 	sb_arctan

y_pos:
	sub $a0, $t5, $t1  #delta x
	sub $a1, $t4, $t2  #delta y
	add $t4, $t4, 3
	blt $t4, 90, sb_arctan
	sub $t4, $t4, 80
	j 	sb_arctan

location:
	lw  $t1, BOT_X
	lw  $t2, BOT_Y
	bge $t1, 150, right_quad
	bge $t2, 150, quad_three
	j		quad_four

right_quad:
	bge $t2, 150, quad_two
	j		quad_one

quad_one:
	lw  $t2, BOT_Y
	bgt $t2, 150, infinite
	j quad_one

quad_two:
	lw  $t1, BOT_X
	blt $t1, 150, infinite
	j quad_two

quad_three:
	lw  $t2, BOT_Y
	blt $t2, 150, infinite
	j quad_three

quad_four:
	lw  $t1, BOT_X
	bgt $t1, 150, infinite
	j quad_four

sb_arctan:
	li	$v0, 0		# angle = 0;

	abs	$t0, $a0	# get absolute values
	abs	$t1, $a1
	ble	$t1, $t0, no_TURN_90

	## if (abs(y) > abs(x)) { rotate 90 degrees }
	move	$t0, $a1	# int temp = y;
	neg	$a1, $a0	# y = -x;
	move	$a0, $t0	# x = temp;
	li	$v0, 90		# angle = 90;

no_TURN_90:
	bgez	$a0, pos_x 	# skip if (x >= 0)

	## if (x < 0)
	add	$v0, $v0, 180	# angle += 180;

pos_x:
	mtc1	$a0, $f0
	mtc1	$a1, $f1
	cvt.s.w $f0, $f0	# convert from ints to floats
	cvt.s.w $f1, $f1

	div.s	$f0, $f1, $f0	# float v = (float) y / (float) x;

	mul.s	$f1, $f0, $f0	# v^^2
	mul.s	$f2, $f1, $f0	# v^^3
	l.s	$f3, three	# load 3.0
	div.s 	$f3, $f2, $f3	# v^^3/3
	sub.s	$f6, $f0, $f3	# v - v^^3/3

	mul.s	$f4, $f1, $f2	# v^^5
	l.s	$f5, five	# load 5.0
	div.s 	$f5, $f4, $f5	# v^^5/5
	add.s	$f6, $f6, $f5	# value = v - v^^3/3 + v^^5/5

	l.s	$f8, PI		# load PI
	div.s	$f6, $f6, $f8	# value / PI
	l.s	$f7, F180	# load 180.0
	mul.s	$f6, $f6, $f7	# 180.0 * value / PI

	cvt.w.s $f6, $f6	# convert "delta" back to integer
	mfc1	$t0, $f6
	add	$v0, $v0, $t0	# angle += delta

	sw  $v0, ANGLE
	sw  $t8, ANGLE_CONTROL

	j 	location



.kdata				# interrupt handler data (separated just for readability)
chunkIH:	.space 8	# space for two registers
non_intrpt_str:	.asciiz "Non-interrupt exception\n"
unhandled_str:	.asciiz "Unhandled interrupt type\n"

.ktext 0x80000180
interrupt_handler:
.set noat
	move	$k1, $at		# Save $at
.set at
	la	$k0, chunkIH
	sw	$a0, 0($k0)		# Get some free registers
	sw	$a1, 4($k0)		# by storing them to a global variable

	mfc0	$k0, $13		# Get Cause register
	srl	$a0, $k0, 2
	and	$a0, $a0, 0xf		# ExcCode field
	bne	$a0, 0, non_intrpt

interrupt_dispatch:			# Interrupt:
	mfc0	$k0, $13		# Get Cause register, again
	beq	$k0, 0, done		# handled all outstanding interrupts

	and	$a0, $k0, BONK_MASK	# is there a bonk interrupt?
	bne	$a0, 0, bonk_interrupt

	and	$a0, $k0, REQUEST_STARCOIN_INT_MASK	# is there a timer interrupt?
	bne	$a0, 0, starcoin_interrupt

	# add dispatch for other interrupt types here.

	li	$v0, PRINT_STRING	# Unhandled interrupt types
	la	$a0, unhandled_str
	syscall
	j	done

bonk_interrupt:
      sw      $a1, BONK_ACK   # acknowledge interrupt
			li	 		$t0, 10
			li		  $t1, 5
			sw      $t1, ANGLE
			sw		  $0, ANGLE_CONTROL
			sw      $t0, VELOCITY
      j       interrupt_dispatch       # see if other interrupts are waiting


starcoin_interrupt:
	sw		$a1, REQUEST_STARCOIN_ACK		# acknowledge interrupt
	sw		$a1, startcoin_data
	j		interrupt_dispatch	# see if other interrupts are waiting

non_intrpt:				# was some non-interrupt
	li	$v0, PRINT_STRING
	la	$a0, non_intrpt_str
	syscall				# print out an error message
	# fall through to done

done:
	la	$k0, chunkIH
	lw	$a0, 0($k0)		# Restore saved registers
	lw	$a1, 4($k0)
.set noat
	move	$at, $k1		# Restore $at
.set at
	eret

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

.align 2
event_horizon_data: .space 90000

.text
main:
	la  $t7, event_horizon_data
	sw  $t7, REQUEST_JETSTREAM
	li  $t1, 10
	sw  $t1, VELOCITY  #set the initial velocity as 4
	li  $t0, 1
	li  $t8, 5
	li  $t3, 270
	sw  $t8, ANGLE
	sw  $t0, ANGLE_CONTROL

loop:
	lw  $t3, BOT_X
	lw  $t4, BOT_Y
	mul $t5, $t4, 300
	add $t5, $t5, $t3
	add $t6, $t7, $t5
	lb  $t5, 0($t6)
	j angle_loop

angle_loop:
	bgt $t3, 150, right
	bgt $t4, 150, left_bot
	j		left_top

right:
	bgt $t4, 150, right_bot
	j		right_top

left_bot:
	beq $t5, 1, left_check
	li  $t2, 275	#270
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	#sw  $t1, VELOCITY  #set the initial velocity as 4
	j loop

left_top:
	beq $t5, 1, up_check
	li  $t2, 5	#0
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	#sw  $t1, VELOCITY  			#set the initial velocity as 4
	j loop

right_bot:
	beq $t5, 1, down_check
	li  $t2, 175	#180
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	#sw  $t1, VELOCITY  			#set the initial velocity as 4
	j loop

right_top:
	beq $t5, 1, right_check
	li  $t2, 85	#90
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	#sw  $t1, VELOCITY  #set the initial velocity as 4
	j loop

up_check:
	#sw  $t8, VELOCITY  #set the initial velocity as 4
	li  $t2, 275
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	j loop

down_check:
	#sw  $t8, VELOCITY  #set the initial velocity as 4
	li  $t2, 85
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	j loop

left_check:
	#sw  $t8, VELOCITY  #set the initial velocity as 4
	li  $t2, 180
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	j loop

right_check:
	#sw  $t8, VELOCITY  #set the initial velocity as 4
	li  $t2, 0
	sw  $t2, ANGLE
	sw  $t0, ANGLE_CONTROL
	j loop

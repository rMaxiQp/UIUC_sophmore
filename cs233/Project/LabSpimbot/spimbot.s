# syscall constants
PRINT_STRING = 4
PRINT_CHAR   = 11
PRINT_INT    = 1

# debug constants
PRINT_INT_ADDR   = 0xffff0080
PRINT_FLOAT_ADDR = 0xffff0084
PRINT_HEX_ADDR   = 0xffff0088

# spimbot memory-mapped I/O
VELOCITY       = 0xffff0010
ANGLE          = 0xffff0014
ANGLE_CONTROL  = 0xffff0018
BOT_X          = 0xffff0020
BOT_Y          = 0xffff0024
OTHER_BOT_X    = 0xffff00a0
OTHER_BOT_Y    = 0xffff00a4
TIMER          = 0xffff001c
SCORES_REQUEST = 0xffff1018

REQUEST_JETSTREAM    = 0xffff00dc
REQUEST_RADAR        = 0xffff00e0
BANANA            = 0xffff0040
MUSHROOM        = 0xffff0044
STARCOIN        = 0xffff0048

REQUEST_PUZZLE        = 0xffff00d0
SUBMIT_SOLUTION        = 0xffff00d4

# interrupt constants
BONK_MASK    = 0x1000
BONK_ACK    = 0xffff0060

TIMER_MASK    = 0x8000
TIMER_ACK    = 0xffff006c

REQUEST_RADAR_INT_MASK    = 0x4000
REQUEST_RADAR_ACK    = 0xffff00e4

REQUEST_PUZZLE_ACK    = 0xffff00d8
REQUEST_PUZZLE_INT_MASK    = 0x800

.data
three:    .float    3.0
five:    .float    5.0
PI:    .float    3.141592
F180:    .float  180.0

.align 2
type: .space 4
flag: .space 4
puzzle_decrypt: .space 64
soln: .space 64
puzzle_data: .space 220
star_data: .space 512
event_horizon: .space 90000
inv_sbox:
.byte 0x52 0x09 0x6A 0xD5 0x30 0x36 0xA5 0x38 0xBF 0x40 0xA3 0x9E 0x81 0xF3 0xD7 0xFB
.byte 0x7C 0xE3 0x39 0x82 0x9B 0x2F 0xFF 0x87 0x34 0x8E 0x43 0x44 0xC4 0xDE 0xE9 0xCB
.byte 0x54 0x7B 0x94 0x32 0xA6 0xC2 0x23 0x3D 0xEE 0x4C 0x95 0x0B 0x42 0xFA 0xC3 0x4E
.byte 0x08 0x2E 0xA1 0x66 0x28 0xD9 0x24 0xB2 0x76 0x5B 0xA2 0x49 0x6D 0x8B 0xD1 0x25
.byte 0x72 0xF8 0xF6 0x64 0x86 0x68 0x98 0x16 0xD4 0xA4 0x5C 0xCC 0x5D 0x65 0xB6 0x92
.byte 0x6C 0x70 0x48 0x50 0xFD 0xED 0xB9 0xDA 0x5E 0x15 0x46 0x57 0xA7 0x8D 0x9D 0x84
.byte 0x90 0xD8 0xAB 0x00 0x8C 0xBC 0xD3 0x0A 0xF7 0xE4 0x58 0x05 0xB8 0xB3 0x45 0x06
.byte 0xD0 0x2C 0x1E 0x8F 0xCA 0x3F 0x0F 0x02 0xC1 0xAF 0xBD 0x03 0x01 0x13 0x8A 0x6B
.byte 0x3A 0x91 0x11 0x41 0x4F 0x67 0xDC 0xEA 0x97 0xF2 0xCF 0xCE 0xF0 0xB4 0xE6 0x73
.byte 0x96 0xAC 0x74 0x22 0xE7 0xAD 0x35 0x85 0xE2 0xF9 0x37 0xE8 0x1C 0x75 0xDF 0x6E
.byte 0x47 0xF1 0x1A 0x71 0x1D 0x29 0xC5 0x89 0x6F 0xB7 0x62 0x0E 0xAA 0x18 0xBE 0x1B
.byte 0xFC 0x56 0x3E 0x4B 0xC6 0xD2 0x79 0x20 0x9A 0xDB 0xC0 0xFE 0x78 0xCD 0x5A 0xF4
.byte 0x1F 0xDD 0xA8 0x33 0x88 0x07 0xC7 0x31 0xB1 0x12 0x10 0x59 0x27 0x80 0xEC 0x5F
.byte 0x60 0x51 0x7F 0xA9 0x19 0xB5 0x4A 0x0D 0x2D 0xE5 0x7A 0x9F 0x93 0xC9 0x9C 0xEF
.byte 0xA0 0xE0 0x3B 0x4D 0xAE 0x2A 0xF5 0xB0 0xC8 0xEB 0xBB 0x3C 0x83 0x53 0x99 0x61
.byte 0x17 0x2B 0x04 0x7E 0xBA 0x77 0xD6 0x26 0xE1 0x69 0x14 0x63 0x55 0x21 0x0C 0x7D

inv_mix:
.byte 0x00 0x0e 0x1c 0x12 0x38 0x36 0x24 0x2a 0x70 0x7e 0x6c 0x62 0x48 0x46 0x54 0x5a
.byte 0xe0 0xee 0xfc 0xf2 0xd8 0xd6 0xc4 0xca 0x90 0x9e 0x8c 0x82 0xa8 0xa6 0xb4 0xba
.byte 0xdb 0xd5 0xc7 0xc9 0xe3 0xed 0xff 0xf1 0xab 0xa5 0xb7 0xb9 0x93 0x9d 0x8f 0x81
.byte 0x3b 0x35 0x27 0x29 0x03 0x0d 0x1f 0x11 0x4b 0x45 0x57 0x59 0x73 0x7d 0x6f 0x61
.byte 0xad 0xa3 0xb1 0xbf 0x95 0x9b 0x89 0x87 0xdd 0xd3 0xc1 0xcf 0xe5 0xeb 0xf9 0xf7
.byte 0x4d 0x43 0x51 0x5f 0x75 0x7b 0x69 0x67 0x3d 0x33 0x21 0x2f 0x05 0x0b 0x19 0x17
.byte 0x76 0x78 0x6a 0x64 0x4e 0x40 0x52 0x5c 0x06 0x08 0x1a 0x14 0x3e 0x30 0x22 0x2c
.byte 0x96 0x98 0x8a 0x84 0xae 0xa0 0xb2 0xbc 0xe6 0xe8 0xfa 0xf4 0xde 0xd0 0xc2 0xcc
.byte 0x41 0x4f 0x5d 0x53 0x79 0x77 0x65 0x6b 0x31 0x3f 0x2d 0x23 0x09 0x07 0x15 0x1b
.byte 0xa1 0xaf 0xbd 0xb3 0x99 0x97 0x85 0x8b 0xd1 0xdf 0xcd 0xc3 0xe9 0xe7 0xf5 0xfb
.byte 0x9a 0x94 0x86 0x88 0xa2 0xac 0xbe 0xb0 0xea 0xe4 0xf6 0xf8 0xd2 0xdc 0xce 0xc0
.byte 0x7a 0x74 0x66 0x68 0x42 0x4c 0x5e 0x50 0x0a 0x04 0x16 0x18 0x32 0x3c 0x2e 0x20
.byte 0xec 0xe2 0xf0 0xfe 0xd4 0xda 0xc8 0xc6 0x9c 0x92 0x80 0x8e 0xa4 0xaa 0xb8 0xb6
.byte 0x0c 0x02 0x10 0x1e 0x34 0x3a 0x28 0x26 0x7c 0x72 0x60 0x6e 0x44 0x4a 0x58 0x56
.byte 0x37 0x39 0x2b 0x25 0x0f 0x01 0x13 0x1d 0x47 0x49 0x5b 0x55 0x7f 0x71 0x63 0x6d
.byte 0xd7 0xd9 0xcb 0xc5 0xef 0xe1 0xf3 0xfd 0xa7 0xa9 0xbb 0xb5 0x9f 0x91 0x83 0x8d
.byte 0x00 0x0b 0x16 0x1d 0x2c 0x27 0x3a 0x31 0x58 0x53 0x4e 0x45 0x74 0x7f 0x62 0x69
.byte 0xb0 0xbb 0xa6 0xad 0x9c 0x97 0x8a 0x81 0xe8 0xe3 0xfe 0xf5 0xc4 0xcf 0xd2 0xd9
.byte 0x7b 0x70 0x6d 0x66 0x57 0x5c 0x41 0x4a 0x23 0x28 0x35 0x3e 0x0f 0x04 0x19 0x12
.byte 0xcb 0xc0 0xdd 0xd6 0xe7 0xec 0xf1 0xfa 0x93 0x98 0x85 0x8e 0xbf 0xb4 0xa9 0xa2
.byte 0xf6 0xfd 0xe0 0xeb 0xda 0xd1 0xcc 0xc7 0xae 0xa5 0xb8 0xb3 0x82 0x89 0x94 0x9f
.byte 0x46 0x4d 0x50 0x5b 0x6a 0x61 0x7c 0x77 0x1e 0x15 0x08 0x03 0x32 0x39 0x24 0x2f
.byte 0x8d 0x86 0x9b 0x90 0xa1 0xaa 0xb7 0xbc 0xd5 0xde 0xc3 0xc8 0xf9 0xf2 0xef 0xe4
.byte 0x3d 0x36 0x2b 0x20 0x11 0x1a 0x07 0x0c 0x65 0x6e 0x73 0x78 0x49 0x42 0x5f 0x54
.byte 0xf7 0xfc 0xe1 0xea 0xdb 0xd0 0xcd 0xc6 0xaf 0xa4 0xb9 0xb2 0x83 0x88 0x95 0x9e
.byte 0x47 0x4c 0x51 0x5a 0x6b 0x60 0x7d 0x76 0x1f 0x14 0x09 0x02 0x33 0x38 0x25 0x2e
.byte 0x8c 0x87 0x9a 0x91 0xa0 0xab 0xb6 0xbd 0xd4 0xdf 0xc2 0xc9 0xf8 0xf3 0xee 0xe5
.byte 0x3c 0x37 0x2a 0x21 0x10 0x1b 0x06 0x0d 0x64 0x6f 0x72 0x79 0x48 0x43 0x5e 0x55
.byte 0x01 0x0a 0x17 0x1c 0x2d 0x26 0x3b 0x30 0x59 0x52 0x4f 0x44 0x75 0x7e 0x63 0x68
.byte 0xb1 0xba 0xa7 0xac 0x9d 0x96 0x8b 0x80 0xe9 0xe2 0xff 0xf4 0xc5 0xce 0xd3 0xd8
.byte 0x7a 0x71 0x6c 0x67 0x56 0x5d 0x40 0x4b 0x22 0x29 0x34 0x3f 0x0e 0x05 0x18 0x13
.byte 0xca 0xc1 0xdc 0xd7 0xe6 0xed 0xf0 0xfb 0x92 0x99 0x84 0x8f 0xbe 0xb5 0xa8 0xa3
.byte 0x00 0x0d 0x1a 0x17 0x34 0x39 0x2e 0x23 0x68 0x65 0x72 0x7f 0x5c 0x51 0x46 0x4b
.byte 0xd0 0xdd 0xca 0xc7 0xe4 0xe9 0xfe 0xf3 0xb8 0xb5 0xa2 0xaf 0x8c 0x81 0x96 0x9b
.byte 0xbb 0xb6 0xa1 0xac 0x8f 0x82 0x95 0x98 0xd3 0xde 0xc9 0xc4 0xe7 0xea 0xfd 0xf0
.byte 0x6b 0x66 0x71 0x7c 0x5f 0x52 0x45 0x48 0x03 0x0e 0x19 0x14 0x37 0x3a 0x2d 0x20
.byte 0x6d 0x60 0x77 0x7a 0x59 0x54 0x43 0x4e 0x05 0x08 0x1f 0x12 0x31 0x3c 0x2b 0x26
.byte 0xbd 0xb0 0xa7 0xaa 0x89 0x84 0x93 0x9e 0xd5 0xd8 0xcf 0xc2 0xe1 0xec 0xfb 0xf6
.byte 0xd6 0xdb 0xcc 0xc1 0xe2 0xef 0xf8 0xf5 0xbe 0xb3 0xa4 0xa9 0x8a 0x87 0x90 0x9d
.byte 0x06 0x0b 0x1c 0x11 0x32 0x3f 0x28 0x25 0x6e 0x63 0x74 0x79 0x5a 0x57 0x40 0x4d
.byte 0xda 0xd7 0xc0 0xcd 0xee 0xe3 0xf4 0xf9 0xb2 0xbf 0xa8 0xa5 0x86 0x8b 0x9c 0x91
.byte 0x0a 0x07 0x10 0x1d 0x3e 0x33 0x24 0x29 0x62 0x6f 0x78 0x75 0x56 0x5b 0x4c 0x41
.byte 0x61 0x6c 0x7b 0x76 0x55 0x58 0x4f 0x42 0x09 0x04 0x13 0x1e 0x3d 0x30 0x27 0x2a
.byte 0xb1 0xbc 0xab 0xa6 0x85 0x88 0x9f 0x92 0xd9 0xd4 0xc3 0xce 0xed 0xe0 0xf7 0xfa
.byte 0xb7 0xba 0xad 0xa0 0x83 0x8e 0x99 0x94 0xdf 0xd2 0xc5 0xc8 0xeb 0xe6 0xf1 0xfc
.byte 0x67 0x6a 0x7d 0x70 0x53 0x5e 0x49 0x44 0x0f 0x02 0x15 0x18 0x3b 0x36 0x21 0x2c
.byte 0x0c 0x01 0x16 0x1b 0x38 0x35 0x22 0x2f 0x64 0x69 0x7e 0x73 0x50 0x5d 0x4a 0x47
.byte 0xdc 0xd1 0xc6 0xcb 0xe8 0xe5 0xf2 0xff 0xb4 0xb9 0xae 0xa3 0x80 0x8d 0x9a 0x97
.byte 0x00 0x09 0x12 0x1b 0x24 0x2d 0x36 0x3f 0x48 0x41 0x5a 0x53 0x6c 0x65 0x7e 0x77
.byte 0x90 0x99 0x82 0x8b 0xb4 0xbd 0xa6 0xaf 0xd8 0xd1 0xca 0xc3 0xfc 0xf5 0xee 0xe7
.byte 0x3b 0x32 0x29 0x20 0x1f 0x16 0x0d 0x04 0x73 0x7a 0x61 0x68 0x57 0x5e 0x45 0x4c
.byte 0xab 0xa2 0xb9 0xb0 0x8f 0x86 0x9d 0x94 0xe3 0xea 0xf1 0xf8 0xc7 0xce 0xd5 0xdc
.byte 0x76 0x7f 0x64 0x6d 0x52 0x5b 0x40 0x49 0x3e 0x37 0x2c 0x25 0x1a 0x13 0x08 0x01
.byte 0xe6 0xef 0xf4 0xfd 0xc2 0xcb 0xd0 0xd9 0xae 0xa7 0xbc 0xb5 0x8a 0x83 0x98 0x91
.byte 0x4d 0x44 0x5f 0x56 0x69 0x60 0x7b 0x72 0x05 0x0c 0x17 0x1e 0x21 0x28 0x33 0x3a
.byte 0xdd 0xd4 0xcf 0xc6 0xf9 0xf0 0xeb 0xe2 0x95 0x9c 0x87 0x8e 0xb1 0xb8 0xa3 0xaa
.byte 0xec 0xe5 0xfe 0xf7 0xc8 0xc1 0xda 0xd3 0xa4 0xad 0xb6 0xbf 0x80 0x89 0x92 0x9b
.byte 0x7c 0x75 0x6e 0x67 0x58 0x51 0x4a 0x43 0x34 0x3d 0x26 0x2f 0x10 0x19 0x02 0x0b
.byte 0xd7 0xde 0xc5 0xcc 0xf3 0xfa 0xe1 0xe8 0x9f 0x96 0x8d 0x84 0xbb 0xb2 0xa9 0xa0
.byte 0x47 0x4e 0x55 0x5c 0x63 0x6a 0x71 0x78 0x0f 0x06 0x1d 0x14 0x2b 0x22 0x39 0x30
.byte 0x9a 0x93 0x88 0x81 0xbe 0xb7 0xac 0xa5 0xd2 0xdb 0xc0 0xc9 0xf6 0xff 0xe4 0xed
.byte 0x0a 0x03 0x18 0x11 0x2e 0x27 0x3c 0x35 0x42 0x4b 0x50 0x59 0x66 0x6f 0x74 0x7d
.byte 0xa1 0xa8 0xb3 0xba 0x85 0x8c 0x97 0x9e 0xe9 0xe0 0xfb 0xf2 0xcd 0xc4 0xdf 0xd6
.byte 0x31 0x38 0x23 0x2a 0x15 0x1c 0x07 0x0e 0x79 0x70 0x6b 0x62 0x5d 0x54 0x4f 0x46

.text
main:
  #enable all interrupt
  la     $t4, type
  lw     $t4, OTHER_BOT_X
  sw     $t4, type

  li     $t4, REQUEST_RADAR_INT_MASK
  or       $t4, $t4, TIMER_MASK    # bonk interrupt bit
  or     $t4, $t4, REQUEST_PUZZLE_INT_MASK
  or     $t4, $t4, 1
  mtc0   $t4, $12
  la     $s4, soln
  la     $t2, star_data
  la     $t0, event_horizon
  la     $t4, puzzle_data
  sw     $t0, REQUEST_JETSTREAM # jetstream
  li     $t5, 10
  sw     $t5, VELOCITY  # set VELOCITY
  li     $s0, 1         # constant number for ANGLE_CONTROL
  sw     $s0, flag
  lw       $v0, TIMER        # current time
  add       $v0, $v0, 500
  sw       $v0, TIMER        # request timer in 20000 cycles
  li     $t5, 0
  li     $t6, 0

for_loop_y:
    mul $s2, $t6, 300
    add $s2, $s2, 150
    add $s2, $t0, $s2
    lb  $s2, 0($s2)
    beq $s2, 2, for_loop_x
    add $t6, $t6, 1
    j for_loop_y

for_loop_x:
    add $s2, $t5, 45000
    add $s2, $t0, $s2
    lb  $s2, 0($s2)
    beq $s2, 2, calculate
    add $t5, $t5, 1
    j for_loop_x
x_greater:
    sub $t6, $t6, 149
    sub $t7, $0, $t6
    li  $t5, 0
    li  $t0, 0
    j loop

y_greater:
    sub $t5, $t5, 149
    sub $t7, $0, $t5
    li  $t5, 0
    li  $t0, 0
    j loop

calculate:
    blt  $t5, $t6, x_greater

    j y_greater

#     move $a0, $0
#     sub  $a1, $t5, 150
#     jal  euclidean_dist
#     move $t7, $v0
#     sub  $a1, $t6, 150
#     jal  euclidean_dist
#     add  $t7, $v0, $t7
#     srl  $t7, $t7, 1

# for_loop_y:
#     mul $s2, $t6, 300
#     add $s2, $s2, 75
#     add $s2, $t0, $s2
#     lb  $s2, 0($s2)
#     beq $s2, 2, for_loop_x
#     add $t6, $t6, 1
#     j for_loop_y
#
# for_loop_x:
#     add $s2, $t5, 22500
#     add $s2, $t0, $s2
#     lb  $s2, 0($s2)
#     beq $s2, 2, calculate
#     add $t5, $t5, 1
#     j for_loop_x
#
# calculate:
#     sub  $a0, $t5, 150
#     sub  $a1, $t6, 150
#     jal  euclidean_dist
#     move $t7, $a0

#  4th | 1st
#  ----|----
#  3rd | 2nd

loop:
  beq    $t5, 1, solve
  lw     $t0, STARCOIN
  #sw     $t2, REQUEST_RADAR
  blt    $t0, 4, cc
  sw     $t4, REQUEST_PUZZLE

cc:
#  bne    $t2, $0, find
  li     $t8, 10
  sw     $t8, VELOCITY

  lw     $t8, BOT_X
  lw     $t9, BOT_Y
  sub    $a0,  $t8, 150
    move   $t8,  $a0
  sub    $a1,  $t9, 150
    move   $t9,  $a1
  jal      sb_arctan
    move   $t3, $v0      # the angle toward the center
  move   $a0, $t8
  move   $a1, $t9
    jal    euclidean_dist
    bge    $t7, $v0, inn # the bot is moving toward the center
    bge    $t8, 0, right
    bge    $t9, 0, quad_2
    add    $t6, $t3, 90
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

# find:
#   add    $t4, $t4, 4
#   lw     $t0, 0($t4)
#   bne    $t0, 0xffffffff, find
#   add    $t4, $t4, 4
#   lw     $t0, 0($t4)
#   srl    $t5, $t0, 16
#   sll    $t6, $t0, 16
#   srl    $t6, $t6, 16
#   sub    $a0, $t8, $t5
#   sub    $a1, $t9, $t6
#   jal    euclidean_dist
#   bge    $v0, 10, loop
#   lw     $t4, ANGLE
#   add    $t4, $t4, 6
#   sw     $t4, ANGLE
#   li     $t0, 1
#   sw     $t0, ANGLE_CONTROL
#   li     $t4, 0
# halt:
#   add    $t4, $t4, 1
#   blt    $t4, 10000, halt

right:
    bge    $t9, 0, quad_2
    sub    $t6, $t3, 250
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

quad_2:
    add    $t6, $t3, 110
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

inn:
    bge    $t8, 0, quad_1 #BOT_X > 150
    add    $t6, $t3, 70
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

quad_1:
    bge    $t9, 0, rightdown1
    sub    $t6, $t3, 290
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

rightdown1:
    addi   $t6, $t3, 70
    sw       $t6, ANGLE
    sw       $s0, ANGLE_CONTROL
    j      loop

solve:
  sub    $sp, $sp, 52
  sw     $t0, 0($sp)
  sw     $t1, 4($sp)
  sw     $t2, 8($sp)
  sw     $t3, 12($sp)
  sw     $t4, 16($sp)
  sw     $t5, 20($sp)
  sw     $t6, 24($sp)
  sw     $t7, 28($sp)
  sw     $t8, 32($sp)
  sw     $t9, 36($sp)
  sw     $a0, 40($sp)
  sw     $a1, 44($sp)
  sw     $a2, 48($sp)

  li     $t0, 0
  sw     $t0, flag

  add    $a0, $t4, 0 # $a0 for encrypted
  add    $a1, $s4, 0 # $a1 for plaintext
  addi   $a2, $t4, 64 # a2 for key
  addi   $a3, $t4, 208 #a3 for rounds
  lbu    $a3, 0($a3)
  jal    decrypt #first

  lw     $t4, 16($sp)
  add    $a1, $s4, 16
  add    $a0, $t4, 16
  addi   $a2, $t4, 64 # a2 for key
  addi   $a3, $t4, 208 #a3 for rounds
  lbu    $a3, 0($a3)
  jal    decrypt #second

  lw     $t4, 16($sp)
  add    $a1, $s4, 32
  add    $a0, $t4, 32
  addi   $a2, $t4, 64 # a2 for key
  addi   $a3, $t4, 208 #a3 for rounds
  lbu    $a3, 0($a3)
  jal    decrypt #third

  lw     $t4, 16($sp)
  add    $a1, $s4, 48
  add    $a0, $t4, 48
  addi   $a2, $t4, 64 # a2 for key
  addi   $a3, $t4, 208 #a3 for rounds
  lbu    $a3, 0($a3)
  jal    decrypt #fourth

  move   $a0, $s4
  la     $a1, puzzle_decrypt
  lw     $t4, 16($sp)
  lw     $a2, 212($t4)
  jal    max_unique_n_substr
  sw     $a1, SUBMIT_SOLUTION

  lw     $t0, 0($sp)
  lw     $t1, 4($sp)
  lw     $t2, 8($sp)
  lw     $t3, 12($sp)
  lw     $t4, 16($sp)
  lw     $t5, 20($sp)
  lw     $t6, 24($sp)
  lw     $t7, 28($sp)
  lw     $t8, 32($sp)
  lw     $t9, 36($sp)
  lw     $a0, 40($sp)
  lw     $a1, 44($sp)
  lw     $a2, 48($sp)
  add    $sp, $sp, 52
  li     $t5, 0
  sw     $s0, flag
  j loop

#################################
# function that calculate angle #
#################################

.text
sb_arctan:
    li    $v0, 0        # angle = 0;

    abs    $t0, $a0    # get absolute values
    abs    $t1, $a1
    ble    $t1, $t0, no_TURN_90

    ## if (abs(y) > abs(x)) { rotate 90 degrees }
    move    $t0, $a1    # int temp = y;
    neg    $a1, $a0    # y = -x;
    move    $a0, $t0    # x = temp;
    li    $v0, 90        # angle = 90;

no_TURN_90:
    bgez    $a0, pos_x     # skip if (x >= 0)

    ## if (x < 0)
    add    $v0, $v0, 180    # angle += 180;

pos_x:
    mtc1    $a0, $f0
    mtc1    $a1, $f1
    cvt.s.w $f0, $f0    # convert from ints to floats
    cvt.s.w $f1, $f1

    div.s    $f0, $f1, $f0    # float v = (float) y / (float) x;
  l.s    $f3 , three
    mul.s    $f1, $f0, $f0    # v^^2
    mul.s    $f2, $f1, $f0    # v^^3

    div.s     $f3, $f2, $f3    # v^^3/3
    sub.s    $f6, $f0, $f3    # v - v^^3/3

    mul.s    $f4, $f1, $f2    # v^^5
    l.s    $f5, five    # load 5.0
    div.s     $f5, $f4, $f5    # v^^5/5
    add.s    $f6, $f6, $f5    # value = v - v^^3/3 + v^^5/5

    l.s    $f8, PI        # load PI
    div.s    $f6, $f6, $f8    # value / PI
    l.s    $f7, F180    # load 180.0
    mul.s    $f6, $f6, $f7    # 180.0 * value / PI

    cvt.w.s $f6, $f6    # convert "delta" back to integer
    mfc1    $t0, $f6
    add    $v0, $v0, $t0    # angle += delta

    jr     $ra

#####################################
# function that calculate distance  #
#####################################
.text
.globl euclidean_dist
euclidean_dist:
    mul        $a0, $a0, $a0    # x^2
    mul        $a1, $a1, $a1    # y^2
    add        $v0, $a0, $a1    # x^2 + y^2
    mtc1      $v0, $f0
    cvt.s.w    $f0, $f0    # float(x^2 + y^2)
    sqrt.s    $f0, $f0    # sqrt(x^2 + y^2)
    cvt.w.s    $f0, $f0    # int(sqrt(...))
    mfc1      $v0, $f0
    jr        $ra

#####################################
# function of inv_byte_substitution #
#####################################

## void
## inv_byte_substitution(unsigned char *in, unsigned char *out) {
##     for (unsigned int i = 0; i < 16; i++) {
##         out[i] = inv_sbox[in[i]];
##     }
##     return;
## }

.text
.globl inv_byte_substitution
inv_byte_substitution:
    la    $t9, inv_sbox        # $t9 = inv_sbox = &inv_sbox[0]

    move    $t0, $0            # $t0 = unsigned int i = 0
ibs_for:
    bge    $t0, 16, ibs_done    # if (i >= 16), done

    add    $t1, $a0, $t0        # &in[i]
    lbu    $t1, 0($t1)        # in[i]

    add    $t2, $t9, $t1        # &inv_sbox[in[i]]
    lbu    $t2, 0($t2)        # inv_sbox[in[i]]

    add    $t3, $a1, $t0        # &out[i]
    sb    $t2, 0($t3)        # out[i] = inv_sbox[in[i]]

    add    $t0, $t0, 1        # i++
    j    ibs_for

ibs_done:
    jr    $ra

##############################
# function of circular_shift #
##############################

## unsigned int
## circular_shift(unsigned int in, unsigned char s) {
##     return (in >> 8 * s) | (in << (32 - 8 * s));
## }
.text
.globl circular_shift
circular_shift:
    mul    $t0, $a1, 8     # $t0 = 8 * s
    li    $t1, 32
    sub    $t1, $t1, $t0    # t1 = 32 - 8 * s

    srl    $t2, $a0, $t0    # $t2 = in >> 8 * s
    sll    $t3, $a0, $t1    # $t3 = in << (32 - 8 * s)

    or    $v0, $t2, $t3    # (in >> 8 * s) | (in << (32 - 8 * s))
    jr    $ra


############################
# function of key_addition #
############################

## void
## key_addition(unsigned char *in_one, unsigned char *in_two, unsigned char *out) {
##     for (unsigned int i = 0; i < 16; i++) {
##         out[i] = in_one[i] ^ in_two[i];
##     }
## }
.text
.globl key_addition
key_addition:
    move    $t0, $0            # $t0 = unsigned int i = 0
ka_for:
    bge    $t0, 16, ka_done     # if (i >= 16), done

    add    $t1, $a0, $t0        # &in_one[i]
    lbu    $t1, 0($t1)        # in_one[i]

    add    $t2, $a1, $t0        # &in_two[i]
    lbu    $t2, 0($t2)        # in_two[i]

    xor    $t3, $t1, $t2        # in_one[i] ^ in_two[i]

    add    $t4, $a2, $t0        # &out[i]
    sb    $t3, 0($t4)        # out[i] = in_one[i] ^ in_two[i]

    add    $t0, $t0, 1        # i++
    j    ka_for

ka_done:
    jr    $ra

###################################
# function of max_unique_n_substr #
###################################

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
.text
.globl max_unique_n_substr
max_unique_n_substr:
    beq    $a0, $0, muns_abort         # !in_str
    beq    $a1, $0, muns_abort        # !out_str
    beq    $a2, $0, muns_abort        # !n
    j    muns_do

muns_abort:
    jr    $ra

muns_do:
    sub    $sp, $sp, 36
    sw    $ra, 0($sp)
    sw    $s0, 4($sp)            # $s0 = char *in_str
    sw    $s1, 8($sp)            # $s1 = char *out_str
    sw    $s2, 12($sp)            # $s2 = int n
    sw    $s3, 16($sp)            # $s3 = char *max_marker
    sw    $s4, 20($sp)            # $s4 = unsigned int len_max
    sw    $s5, 24($sp)            # $s5 = unsigned int len_in_str
    sw    $s6, 28($sp)            # $s6 = unsigned int cur_pos
    sw    $s7, 32($sp)            # $s7 = int len_cur

    move    $s0, $a0
    move    $s1, $a1
    move    $s2, $a2

    move    $s3, $a0            # max_marker = in_str
    li    $s4, 0                # len_max = 0

    jal    my_strlen            # my_strlen(in_str)
    move    $s5, $v0            # len_in_str = my_strlen(in_str)

    li    $s6, 0                # cur_pos = 0
muns_for:
    bge    $s6, $s5, muns_for_end         # if (cur_pos >= len_in_str), end

    add    $s7, $s0, $s6            # i = in_str + cur_pos

    move    $a0, $s7
    add    $a1, $s2, 1
    jal    nth_uniq_char            # nth_uniq_char(i, n + 1)

    ble    $v0, $s4, muns_for_cont        # if (len_cur <= len_max), continue
    move    $s4, $v0            # len_max = len_cur
    move    $s3, $s7            # max_marker = i

muns_for_cont:
    add    $s6, $s6, 1            # cur_pos++
    j    muns_for

muns_for_end:
    ## Setup call to my_strncpy
    move    $a0, $s1
    move    $a1, $s3
    move    $a2, $s4

    lw      $ra, 0($sp)
    lw      $s0, 4($sp)
    lw      $s1, 8($sp)
    lw      $s2, 12($sp)
    lw      $s3, 16($sp)
    lw      $s4, 20($sp)
    lw      $s5, 24($sp)
    lw      $s6, 28($sp)
    lw      $s7, 32($sp)
    add    $sp, $sp, 36

    ## Tail call
    j    my_strncpy            # my_strncpy(out_str, max_marker, len_max)


  .text

  .globl my_strncpy
  my_strncpy:
      sub    $sp, $sp, 16
      sw    $s0, 0($sp)
      sw    $s1, 4($sp)
      sw    $s2, 8($sp)
      sw    $ra, 12($sp)
      move    $s0, $a0
      move    $s1, $a1
      move    $s2, $a2

      move    $a0, $a1
      jal    my_strlen
      add    $v0, $v0, 1
      bge    $s2, $v0, my_strncpy_if
      move    $v0, $s2
  my_strncpy_if:
      li    $t0, 0
  my_strncpy_for:
      bge    $t0, $v0, my_strncpy_end
      add    $t1, $s1, $t0
      lb    $t2, 0($t1)
      add    $t1, $s0, $t0
      sb    $t2, 0($t1)
      add    $t0, $t0, 1
      j    my_strncpy_for
  my_strncpy_end:
      lw    $s0, 0($sp)
      lw    $s1, 4($sp)
      lw    $s2, 8($sp)
      lw    $ra, 12($sp)
      add    $sp, $sp, 16
      jr    $ra


#########################
# function of mt_strlen #
#########################

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
.text
.globl my_strlen
my_strlen:
    move    $v0, $0            # $v0 = unsigned int count = 0
    bne    $a0, $0, ms_while    # if (in != NULL), skip
    jr    $ra            # return 0

ms_while:
    lb    $t0, 0($a0)        # $t0 = *in
    beq    $t0, $0, ms_done    # if (in == 0), done

    add    $v0, $v0, 1        # count++
    add    $a0, $a0, 1        # in++
    j    ms_while

ms_done:
    jr    $ra            # return count


##########################
# function of uniq_chars #
##########################
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
.text
.globl nth_uniq_char
nth_uniq_char:
    beq    $a0, $0, nuc_abort         # !in_str
    beq    $a1, $0, nuc_abort        # !n
    j    nuc_ok

nuc_abort:
    li    $v0, -1                # return -1
    jr    $ra

nuc_ok:
    la    $t0, uniq_chars            # $t0 = uniq_chars = &uniq_chars[0]
    lb    $t9, 0($a0)            # *in_str
    sb    $t9, 0($t0)            # uniq_chars[0] = *in_str

    li    $t1, 1                # $t1 = int uniq_so_far = 1
    li    $t2, 0                # $t2 = int position = 0
    add    $a0, $a0, 1            # in_str++

nuc_while:
    bge    $t1, $a1, nuc_while_end     # if (uniq_so_far >= n), end
    lb    $t9, 0($a0)            # *in_str
    beq    $t9, $0, nuc_while_end        # if (*in_str == 0), end

    li    $t3, 1                # $t3 = char is_uniq = 1

    li    $t4, 0                # $t4 = int j = 0
nuc_for:
    bge    $t4, $t1, nuc_for_end

    add    $t9, $t0, $t4            # &uniq_chars[j]
    lb    $t9, 0($t9)            # uniq_chars[j]
    lb    $t8, 0($a0)            # *in_str
    bne    $t9, $t8, nuc_for_cont        # if (uniq_chars[j] != *in_str), skip
    li    $t3, 0
    j    nuc_for_end

nuc_for_cont:
    add    $t4, $t4, 1            # j++
    j    nuc_for

nuc_for_end:
    beq    $t3, $0, nuc_while_cont     # if (is_uniq == 0), continue
    lb    $t9, 0($a0)            # *in_str
    add    $t8, $t0, $t1            # &uniq_chars[uniq_so_far]
    sb    $t9, 0($t8)            # uniq_chars[uniq_so_far] = *in_str
    add    $t1, $t1, 1            # uniq_so_far++

nuc_while_cont:
    add    $t2, $t2, 1            # position++
    add    $a0, $a0, 1            # in_str++
    j    nuc_while

nuc_while_end:
    bge    $t1, $a1, nuc_end         # if (uniq_so_far >= n), skip
    add    $t2, $t2, 1            # position++

nuc_end:
    move    $v0, $t2            # return position
    jr    $ra

#######################
# function of decrypt #
#######################

##void decrypt(uint8_t *ciphertext, uint8_t *plaintext, uint8_t *key, uint8_t rounds){
##    uint8_t A[16], B[16], C[16], D[16];
##    key_addition(ciphertext, &key[16 * rounds], C);
##    inv_shift_rows((uint32_t *)C, (uint32_t *)B);
##    inv_byte_substitution(B, A);
##    for (uint32_t k = rounds - 1; k > 0; k--){
##        key_addition(A, &key[16 * k], D);
##        inv_mix_column(D, C);
##        inv_shift_rows((uint32_t *)C, (uint32_t *)B);
##        inv_byte_substitution(B, A);
##    }
##    key_addition(A, key, plaintext);
##    return;
##}
.text
.globl decrypt
decrypt:
  #There is the stack mem and the saved reg
  sub $sp, $sp, 100
     sw    $ra, 0($sp)
    sw    $s0, 4($sp)
    sw    $s1, 8($sp)
    sw    $s2, 12($sp)
    sw    $s3, 16($sp)
    sw    $s4, 20($sp)
    sw    $s5, 24($sp)
    sw    $s6, 28($sp)
    sw    $s7, 32($sp)


  #Args, except rounds
  move $s0, $a0
  move $s1, $a1
  move $s2, $a2
  #stored in s7
  move $s7, $a3

  #A,B,C D loc
  add $s3, $sp, 36
  add $s4, $sp, 52
  add $s5, $sp, 68
  add $s6, $sp, 84

  move $a0, $s0
  mul $t0, $s7,16
  add $a1,$s2 ,$t0
  move $a2, $s5
  jal key_addition

  move $a0, $s5
  move $a1, $s4
  jal inv_shift_rows

  move $a0,$s4
  move $a1,$s3
  jal inv_byte_substitution

  #Rounds - 1
  sub $s7, $s7, 1
for_loop:
  ble $s7, 0,end_for_loop

  move $a0, $s3
  mul $t0, $s7,16
  add $a1, $s2,$t0
  move $a2, $s6
  jal key_addition

  move $a0, $s6
  move $a1, $s5
  jal inv_mix_column

  move $a0, $s5
  move $a1, $s4
  jal inv_shift_rows

  move $a0,$s4
  move $a1,$s3
  jal inv_byte_substitution

  sub $s7, $s7, 1
  j for_loop
end_for_loop:

  move $a0, $s3
  move $a1, $s2
  move $a2, $s1
  jal key_addition

     lw    $ra, 0($sp)
    lw    $s0, 4($sp)
    lw    $s1, 8($sp)
    lw    $s2, 12($sp)
    lw    $s3, 16($sp)
    lw    $s4, 20($sp)
    lw    $s5, 24($sp)
    lw    $s6, 28($sp)
    lw    $s7, 32($sp)
  add $sp, $sp, 100

  jr $ra

##############################
# function of inv_shift_rows #
##############################

  .globl inv_shift_rows
  inv_shift_rows:
      #7 saved registers, 20 for stack
      sub    $sp, $sp, 36
      sw    $ra, 0($sp)
      sw    $s0, 4($sp)
      sw    $s1, 8($sp)
      sw    $s2, 12($sp)
      sw    $s3, 16($sp)

      #Assign M
      add    $s0, $sp, 20
      #Assign in
      move    $s1, $a0
      #assign out
      move    $s2, $a1

      move    $a0, $s1
      move    $a1, $s0
      jal    rearrange_matrix

      #Assign I
      move    $s3, $zero
  for_loop_inv:
      bge    $s3, 4, end_for

      li    $a1, 4
      sub    $a1, $a1, $s3

      mul    $t0, $s3, 4
      add    $t0, $s0, $t0

      lw    $a0, 0($t0)
      jal    circular_shift

      mul    $t0, $s3, 4
      add    $t0, $s0, $t0
      sw    $v0, 0($t0)

      add    $s3, $s3, 1
      j    for_loop_inv
  end_for:
      move    $a0, $s0
      move    $a1, $s2
      jal    rearrange_matrix

      lw    $ra, 0($sp)
      lw    $s0, 4($sp)
      lw    $s1, 8($sp)
      lw    $s2, 12($sp)
      lw    $s3, 16($sp)
      add    $sp, $sp, 36
      jr    $ra

##############################
# function of inv_mix_column #
##############################

.globl inv_mix_column
inv_mix_column:
    sub    $sp, $sp, 16
    sw    $s0, 0($sp)
    sw    $s1, 4($sp)
    sw    $s2, 8($sp)
    sw    $s3, 12($sp)

    move    $s0, $zero
for_first:
    bge    $s0, 4, for_first_done
    move    $s1, $zero
for_second:
    bge    $s1, 4, for_second_done

    #store where out[4*k+i] is
    mul    $t0, $s0, 4
    add    $t0, $t0, $s1
    add    $s3, $a1, $t0
    sb    $zero, 0($s3)

    move    $s2, $zero
for_third:
    bge    $s2, 4, for_third_done
    mul    $t0, $s2, 256
    add    $t1, $s1, $s2
    rem    $t1, $t1, 4
    mul    $t2, $s0, 4
    add    $t2, $t2, $t1
    add    $t2, $t2, $a0

    lbu    $t2, 0($t2)

    add    $t0, $t0, $t2
    la    $t4, inv_mix
    add    $t0, $t0, $t4
    lbu    $t0, 0($t0)

    lb    $t5, 0($s3)
    xor    $t5, $t5, $t0
    sb    $t5, 0($s3)

    add    $s2, $s2, 1
    j    for_third
for_third_done:
    add    $s1, $s1, 1
    j    for_second
for_second_done:
    add    $s0, $s0, 1
    j    for_first
for_first_done:
    lw    $s0, 0($sp)
    lw    $s1, 4($sp)
    lw    $s2, 8($sp)
    lw    $s3, 12($sp)
    add    $sp, $sp, 16
    jr    $ra

################################
# function of rearrange_matrix #
################################

.globl rearrange_matrix
rearrange_matrix:
    move    $t0, $zero
rm_for_loop:
    bge    $t0, 4, end_for_loop_m

    #pointer to out
    mul    $t1, $t0, 4
    add    $t1, $t1, $a1

    sw    $zero, 0($t1)

    move    $t2, $zero
second_for_loop:
    #load in
    bge    $t2, 4, end_second_for_loop
    mul    $t3, $t2, 4
    add    $t3, $t3, $t0
    add    $t3, $a0, $t3

    lbu    $t4, 0($t3)
    mul    $t5, $t2 ,8
    sllv    $t4, $t4, $t5

    lw    $t5, 0($t1)
    or    $t5, $t5, $t4
    sw    $t5, 0($t1)

    add    $t2, $t2, 1
    j    second_for_loop

end_second_for_loop:
    add    $t0, $t0, 1
    j    rm_for_loop
end_for_loop_m:
    jr    $ra

#########################
# function of has_cycle #
#########################

##struct Node {
##  int node_id;              // [0, num_nodes)
##  struct Node ** children;  // Always NULL terminated
##};
##
##//Given a pointer to the root of a graph and the number of nodes,
##//returns 1 if the graph contains a cycle and 0 otherwise
##int
##has_cycle(Node * root, int num_nodes) {
##
##  if (!root) {
##    return 0;
##  }
##
##  Node * stack[num_nodes];
##  stack[0] = root;
##  int stack_size = 1;
##
##  int discovered[num_nodes];
##  for (int i = 0; i < num_nodes; i++) {discovered[i] = 0;}
##
##  while (stack_size > 0) {
##
##    Node * node_ptr = stack[--stack_size];
##
##    if (discovered[node_ptr->node_id]) {
##      return 1;
##    }
##    discovered[node_ptr->node_id] = 1;
##
##    for (Node ** edge_ptr = node_ptr->children; *edge_ptr; edge_ptr++) {
##      stack[stack_size++] = *edge_ptr;
##    }
##  }
##  return 0;
##}
.text
.globl has_cycle
has_cycle:
        bnez    $a0, continue_one
        move    $v0, $0
        jr      $ra

continue_one:
        mul     $t0, $a1, 8                     # allocate space for
                                                # stack[num_nodes]
                                                # discovered[num_nodes]
        sub     $sp, $sp, $t0

        mul     $t0, $a1, 4
        add     $t0, $sp, $t0                   # &stack[0]
        sw      $a0, 0($t0)                     # stack[0] = root;
        li      $t0, 1                          # $t0 <- stack_size = 1
        li      $t1, 0                          # $t1 <- i = 0

for_discovered:
        bge     $t1, $a1, for_discovered_end    # i < num_nodes
        mul     $t2, $t1, 4
        add     $t2, $sp, $t2                   # &discovered[i]
        sw      $0, 0($t2)                      # discovered[i] = 0
        add     $t1, $t1, 1                     # i++
        j       for_discovered
for_discovered_end:

while_begin:
        blez    $t0, return_zero_has_cycle      # stack_size > 0
        sub     $t0, $t0, 1                     # stack_size--
        add     $t1, $t0, $a1
        mul     $t1, $t1, 4
        add     $t1, $sp, $t1                   # &stack[stack_size]
        lw      $t1, 0($t1)                     # $t1 <- node_ptr

        lw      $t2, 0($t1)                     # node_ptr->node_id
        mul     $t2, $t2, 4
        add     $t2, $sp, $t2                   # &discovered[node_ptr->node_id]
        lw      $t3, 0($t2)                     # discovered[node_ptr->node_id]
        beqz    $t3, continue_two
        li      $v0, 1                          # return 1
        j       return_has_cycle
continue_two:
        li      $t3, 1
        sw      $t3, 0($t2)                     # discovered[node_ptr->node_id] = 1

        lw      $t2, 4($t1)                     # $t2 <- edge_ptr = node_ptr->children
for_children:
        lw      $t3, 0($t2)                     # *edge_ptr
        beqz    $t3, for_children_end
        add     $t4, $t0, $a1
        mul     $t4, $t4, 4
        add     $t4, $sp, $t4                   # &stack[stack_size]
        sw      $t3, 0($t4)                     # stack[stack_size] = *edge_ptr;
        add     $t0, $t0, 1                     # stack_size++
        add     $t2, $t2, 4                     # edge_ptr++
        j       for_children
for_children_end:

        j       while_begin

return_zero_has_cycle:
        move    $v0, $0
return_has_cycle:
        mul     $t0, $a1, 8                     # deallocate space for
                                                # stack[num_nodes]
                                                # discovered[num_nodes]
        add     $sp, $sp, $t0
    jr    $ra

#########################
# function of max_depth #
#########################

##struct Node {
##        int node_id;  // [0, num_nodes)
##    struct Node ** children; //always NULL terminated
##};
##
##//Given a pointer to the root of a tree, finds and returns the depth of this tree
##
##int max_depth(Node * root)
##{
##    if(root == NULL)
##        return 0;
##    Node * child = root->children[0];
##    int max = 0;
##    while(child != NULL)
##    {
##        int depth = max_depth(child);
##        if(depth > max)
##            max = depth;
##          child++;
##    }
##    return 1 + max;
##}
.text
.globl max_depth
max_depth:
    bne        $a0, 0, max_depth_continue
    li        $v0, 0
    jr        $ra
max_depth_continue:
    sub        $sp, $sp, 16
    sw        $ra, 0($sp)
    sw        $s0, 4($sp)
    sw        $s1, 8($sp)
    sw        $s2, 12($sp)
    li        $s0, 0                #s0: max = 0
    lw        $s1, 4($a0)
max_depth_while:
    lw        $s2, 0($s1)            #s1: child = current->children[0]
    beq        $s2, $0, max_depth_return
    move        $a0, $s2
    jal        max_depth
    move        $t0, $v0            #t0: depth = max_depth(child)
    ble        $t0, $s0, max_depth_not_max
    move        $s0, $t0
max_depth_not_max:
    add        $s1, $s1, 4            # current->children++ - a bit of disconnect
                            # between current C and MIPS code here, but
                            # this works for now...
    j        max_depth_while
max_depth_return:
    addi  $v0, $s0, 1
    lw        $ra, 0($sp)
    lw        $s0, 4($sp)
    lw        $s1, 8($sp)
    lw        $s2, 12($sp)
    add        $sp, $sp, 16
    jr        $ra


##########################
# function of shift_many #
##########################

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
.text
shift_many:
    sub    $sp, $sp, 20
    sw    $ra, 0($sp)
    sw    $s0, 4($sp)
    sw    $s1, 8($sp)
    sw    $s2, 12($sp)
    sw    $s3, 16($sp)

    move    $s0, $a0
    move    $s1, $a1

    move    $s2, $0
sm_for:
    bge    $s2, 4, sm_done

    mul    $t0, $s2, 4
    add    $t0, $t0, $s0
    lw    $s3, 4($t0)
    beq    $s3, $0, sm_cont

    lw    $a0, 0($s0)
    add    $a1, $s2, $s1
    and    $a1, $a1, 3
    jal    circular_shift
    sw    $v0, 0($s3)

sm_cont:
    add    $s2, $s2, 1
    j    sm_for

sm_done:
    lw    $ra, 0($sp)
    lw    $s0, 4($sp)
    lw    $s1, 8($sp)
    lw    $s2, 12($sp)
    lw    $s3, 16($sp)
    add    $sp, $sp, 20
    jr    $ra

#############################
# interruption handler area #
#############################
.kdata                # interrupt handler data (separated just for readability)
chunk:    .space 48    # space for two registers


.ktext 0x80000180


interrupt_handler:
.set noat
    move      $k1, $at        # Save $at
.set at

    la        $k0, chunk
    sw        $a0, 0($k0)        # Get some free registers
    sw        $a1, 4($k0)        # by storing them to a global variable
  sw      $t0, 8($k0)
  sw      $t1, 12($k0)
  sw      $t4, 16($k0)
  sw      $t6, 20($k0)
  sw      $v0, 24($k0)
  la      $t0, flag
  lw      $t0, 0($t0)
  beq     $t0, $0, done

interrupt_dispatch:            # Interrupt:
    mfc0      $k0, $13        # Get Cause register, again
    beq        $k0, $0, done        # handled all outstanding interrupts
    and        $a0, $k0, REQUEST_RADAR_INT_MASK
    bne        $a0, $0, star_interrupt
  and        $a0, $k0, TIMER_MASK
    bne        $a0, $0, timer_interrupt
  and     $a0, $k0, REQUEST_PUZZLE_INT_MASK
  bne     $a0, $0, puzzle_interrupt
  j       done

timer_interrupt:
    sw        $a1, TIMER_ACK        # acknowledge interrupt
    la    $t0, type
    lw    $t0, ($t0)
  bne     $t0, -1 ,dd
  sw      $t2, REQUEST_RADAR
  sw      $t1, MUSHROOM
  lw        $v0, TIMER        # current time
    add        $v0, $v0, 20000
    sw        $v0, TIMER        # request timer in 20000 cycles
    dd:
    j          interrupt_dispatch # see if other interrupts are waiting

puzzle_interrupt:
  sw      $a1, REQUEST_PUZZLE_ACK
  li      $t5, 1
  j       interrupt_dispatch

star_interrupt:
    li      $t0, 10
    sw      $t0, VELOCITY
  sw      $a1, REQUEST_RADAR_ACK   # acknowledge interrupt

  lw      $t0, ($t2)
  li      $t6, 10
  sw      $t6, VELOCITY
  beq     $t0, 0xffffffff, done
  beq     $t0, -1, interrupt_dispatch
  srl     $t4, $t0, 16
  sll     $t6, $t0, 16
  srl     $t6, $t6, 16     # get startcoin position


  lw      $t0, BOT_X
  lw      $t1, BOT_Y
  sub     $a0, $t0, $t4
  sub     $a1, $t1, $t6
  abs     $a0, $a0
  abs     $a1, $a1
  mul        $a0, $a0, $a0    # x^2
  mul        $a1, $a1, $a1    # y^2
  add        $v0, $a0, $a1    # x^2 + y^2
  mtc1      $v0, $f0
  cvt.s.w    $f0, $f0    # float(x^2 + y^2)
  sqrt.s    $f0, $f0    # sqrt(x^2 + y^2)
  cvt.w.s    $f0, $f0    # int(sqrt(...))
  mfc1      $v0, $f0
  bgt     $v0, 10, interrupt_dispatch  #<- bug!!! change this to 20 and u will see the bug.

inloop:
  lw      $t0, BOT_X
  lw      $t1, BOT_Y
  ble     $t4, $t0, comeback
  li      $t0, 0
  li      $t1, 1
  sw      $t0, ANGLE
  sw      $t1, ANGLE_CONTROL
  j       inloop

comeback:
  lw      $t0, BOT_X
  lw      $t1, BOT_Y
  bge     $t4, $t0, onright
  li      $t0, 180
  li      $t1, 1
  sw      $t0, ANGLE
  sw      $t1, ANGLE_CONTROL
  j       comeback

onright:
  lw      $t0, BOT_X
  lw      $t1, BOT_Y
  blt     $t6, $t1, vcom
  li      $t0, 90
  li      $t1, 1
  sw      $t0, ANGLE
  sw      $t1, ANGLE_CONTROL
  j       onright

vcom:
  lw      $t0, BOT_X
  lw      $t1, BOT_Y
  bge     $t6, $t1, interrupt_dispatch
  li      $t0, 270
  li      $t1, 1
    sw      $t0, ANGLE
  sw      $t1, ANGLE_CONTROL
  j       vcom

done:
    la        $k0, chunk
  lw        $a0, 0($k0)        # Get some free registers
    lw        $a1, 4($k0)        # by storing them to a global variable
  lw      $t0, 8($k0)
  lw      $t1, 12($k0)
  lw      $t4, 16($k0)
  lw      $t6, 20($k0)
  lw      $v0, 24($k0)

.set noat
    move      $at, $k1        # Restore $at
.set at
    eret

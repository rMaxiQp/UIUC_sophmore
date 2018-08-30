.data

# Strings for printing purposes
str1_msg1: .asciiz "decrypt(ciphertext_easy, plaintext_easy, key, 1): "
str2_msg1: .asciiz "decrypt(ctext1_easy, ptext1_easy, key, 2): "
str1_msg2: .asciiz "decrypt(ciphertext, plaintext, key, 10): "
str2_msg2: .asciiz "decrypt(ctext1, ptext1, key, 10): "

# Arrays
ciphertext_easy:
.byte 0x76 0x12 0x0d 0x3e 
.byte 0x6b 0x0a 0x9b 0xf1 
.byte 0x06 0x91 0x01 0xb2 
.byte 0x69 0xf0 0xa3 0x6c 
.byte 0x0

plaintext_easy: .space 17

ciphertext: 
.byte 0x61 0xb7 0xdd 0x48 
.byte 0x82 0xe7 0xe3 0xbf 
.byte 0xc7 0xd4 0x43 0x4f 
.byte 0x3c 0xea 0x61 0xdf 
.byte 0x0

plaintext: .space 17

ctext1_easy:
.byte 0xaf 0x2a 0xde 0xe6                            
.byte 0xc1 0x15 0x81 0x31                            
.byte 0x60 0x26 0x72 0xfe                            
.byte 0xef 0x53 0xb1 0x3c                            
.byte 0x0

ptext1_easy: .space 17

ctext1:
.byte 0xf7 0x1b 0x08 0xbb
.byte 0xb1 0x6d 0x37 0x8e
.byte 0x28 0x6e 0x01 0x98
.byte 0xde 0x6a 0xb1 0xab
.byte 0x0

ptext1: .space 17

.align 2
key: 
.byte 0x2b 0x7e 0x15 0x16 
.byte 0x28 0xae 0xd2 0xa6 
.byte 0xab 0xf7 0x15 0x88 
.byte 0x09 0xcf 0x4f 0x3c 
.byte 0xa0 0xfa 0xfe 0x17 
.byte 0x88 0x54 0x2c 0xb1 
.byte 0x23 0xa3 0x39 0x39 
.byte 0x2a 0x6c 0x76 0x05 
.byte 0xf2 0xc2 0x95 0xf2 
.byte 0x7a 0x96 0xb9 0x43 
.byte 0x59 0x35 0x80 0x7a 
.byte 0x73 0x59 0xf6 0x7f 
.byte 0x3d 0x80 0x47 0x7d 
.byte 0x47 0x16 0xfe 0x3e 
.byte 0x1e 0x23 0x7e 0x44 
.byte 0x6d 0x7a 0x88 0x3b 
.byte 0xef 0x44 0xa5 0x41 
.byte 0xa8 0x52 0x5b 0x7f 
.byte 0xb6 0x71 0x25 0x3b 
.byte 0xdb 0x0b 0xad 0x00 
.byte 0xd4 0xd1 0xc6 0xf8 
.byte 0x7c 0x83 0x9d 0x87 
.byte 0xca 0xf2 0xb8 0xbc 
.byte 0x11 0xf9 0x15 0xbc 
.byte 0x6d 0x88 0xa3 0x7a 
.byte 0x11 0x0b 0x3e 0xfd 
.byte 0xdb 0xf9 0x86 0x41 
.byte 0xca 0x00 0x93 0xfd 
.byte 0x4e 0x54 0xf7 0x0e 
.byte 0x5f 0x5f 0xc9 0xf3 
.byte 0x84 0xa6 0x4f 0xb2 
.byte 0x4e 0xa6 0xdc 0x4f 
.byte 0xea 0xd2 0x73 0x21 
.byte 0xb5 0x8d 0xba 0xd2 
.byte 0x31 0x2b 0xf5 0x60 
.byte 0x7f 0x8d 0x29 0x2f 
.byte 0xac 0x77 0x66 0xf3 
.byte 0x19 0xfa 0xdc 0x21 
.byte 0x28 0xd1 0x29 0x41 
.byte 0x57 0x5c 0x00 0x6e 
.byte 0xd0 0x14 0xf9 0xa8 
.byte 0xc9 0xee 0x25 0x89 
.byte 0xe1 0x3f 0x0c 0xc8 
.byte 0xb6 0x63 0x0c 0xa6 


.text
MAIN_STK_SPC = 4
main:
	sub	$sp, $sp, MAIN_STK_SPC
	sw	$ra, 0($sp)

	# ciphertext_easy, plaintext_easy
	la	$a0, str1_msg1
	jal	print_string

	la	$a0, ciphertext_easy
	la	$a1, plaintext_easy
	la	$a2, key
	li	$a3, 1

	jal	decrypt

	la	$a0, plaintext_easy
	jal	print_string
	jal	print_newline

	# ctext1_easy, ptext1_easy
	la	$a0, str2_msg1
	jal	print_string

	la	$a0, ctext1_easy
	la	$a1, ptext1_easy
	la	$a2, key
	li	$a3, 2

	jal	decrypt

	la	$a0, ptext1_easy
	jal	print_string
	jal	print_newline

	# ciphertext, plaintext
	la	$a0, str1_msg2
	jal	print_string

	la	$a0, ciphertext
	la	$a1, plaintext
	la	$a2, key
	li	$a3, 10

	jal	decrypt

	la	$a0, plaintext
	jal	print_string
	jal	print_newline

	# ctext1, ptext1
	la	$a0, str2_msg2
	jal	print_string

	la	$a0, ctext1
	la	$a1, ptext1
	la	$a2, key
	li	$a3, 10

	jal	decrypt

	la	$a0, ptext1
	jal	print_string
	jal	print_newline

	lw	$ra, 0($sp)
	add	$sp, $sp, MAIN_STK_SPC
	jr	$ra
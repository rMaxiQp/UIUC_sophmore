/*
 * CS 233 Fall 2017 Lab 7
 * clang++ Lab7.cpp -o lab7
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

extern unsigned char inv_sbox[];
void my_strncpy(char *dest, char *src, unsigned int n);

// Part 1 Functions

/*
 * Function: my_strlen
 * --------------------
 * Finds the length of a null terminated string.
 *
 * in: input string
 *
 * returns: length of string, not including null terminating character
 */
unsigned int
my_strlen(char *in) {
    if (!in)
        return 0;

    unsigned int count = 0;
    while (*in) {
        count++;
        in++;
    }

    return count;
}

/*
 * Function: circular_shift
 * --------------------
 * Right shifts an integer by s bytes.
 * An an example, say you have an int:
 * 0x01234567.
 * Shifting right one byte would return:
 * 0x67012345.
 *
 * in: input integer
 * s: number of bytes to right shift by (0-3)
 *
 * returns: the circular shifted integer
 */
unsigned int
circular_shift(unsigned int in, unsigned char s) {
    return (in >> 8 * s) | (in << (32 - 8 * s));
}

/*
 * Function: key_addition
 * --------------------
 * Takes in two char arrays of size 16 and
 * XORs them together.
 *
 * in_one: char array of size 16
 * in_two: char array of size 16
 * out: the XOR of the two input char arrays
 */
void
key_addition(unsigned char *in_one, unsigned char *in_two, unsigned char *out) {
    for (unsigned int i = 0; i < 16; i++) {
        out[i] = in_one[i] ^ in_two[i];
    }
}

// Part 2 Functions

/*
 * Function: inv_byte_substitution
 * --------------------
 * Take an input char array and for each
 * byte, assign the corresponding position
 * in out to be the byte defined by the
 * inv_sbox lookup table.
 *
 * in: char array of size 16
 * out: char array of size 16
 */
void
inv_byte_substitution(unsigned char *in, unsigned char *out) {
    for (unsigned int i = 0; i < 16; i++) {
        out[i] = inv_sbox[in[i]];
    }
    return;
}

/*
 * Function: nth_uniq_char
 * --------------------
 * Finds the first instance of the nth unique character in the input string.
 *
 * in_str: input string
 * n: number of unique character values to search for
 *
 * returns: position of first instance of the nth unique character value found
 */
char uniq_chars[256];
int
nth_uniq_char(char *in_str, int n) {
    if (!in_str || !n)
        return -1;

    uniq_chars[0] = *in_str;
    int uniq_so_far = 1;
    int position = 0;
    in_str++;
    while (uniq_so_far < n && *in_str) {
        char is_uniq = 1;
        for (int j = 0; j < uniq_so_far; j++) {
            if (uniq_chars[j] == *in_str) {
                is_uniq = 0;
                break;
            }
        }
        if (is_uniq) {
            uniq_chars[uniq_so_far] = *in_str;
            uniq_so_far++;
        }
        position++;
        in_str++;
    }

    if (uniq_so_far < n) {
        position++;
    }
    return position;
}

/* Function: max_unique_n_substr
 * --------------------
 * Finds a maximum length substring which contains up to n
 * unique characters.
 *
 * in_str: input string
 * out_str: output string in which to place the results
 * n: number of unique characters allowed in out_str
 */
void
max_unique_n_substr(char *in_str, char *out_str, int n) {
    if (!in_str || !out_str || !n)
        return;

    char *max_marker = in_str;
    unsigned int len_max = 0;
    unsigned int len_in_str = my_strlen(in_str);
    for (unsigned int cur_pos = 0; cur_pos < len_in_str; cur_pos++) {
        char *i = in_str + cur_pos;
        int len_cur = nth_uniq_char(i, n + 1);
        if (len_cur > len_max) {
            len_max = len_cur;
            max_marker = i;
        }
    }

    my_strncpy(out_str, max_marker, len_max);
}

/*
 * Unit tests.
 * Feel free to add more if it helps with testing mips.
 */
void
test_my_strlen() {
    char str[] = "Hello world";
    char str1[] = "";
    assert(my_strlen(str) == 11);
    printf("my_strlen(str) expected: %d\nactual: %d\n", 11, my_strlen(str));
    assert(my_strlen(str1) == 0);
    printf("my_strlen(str1) expected: %d\nactual: %d\n", 0, my_strlen(str1));
    assert(my_strlen(NULL) == 0);
    printf("my_strlen(NULL) expected: %d\nactual: %d\n", 0, my_strlen(NULL));
}

void
test_circular_shift() {
    printf("circular_shift(0, 128): %u\n", circular_shift(0, 128));
    printf("circular_shift(2095, 6): %u\n", circular_shift(2095, 6));
    printf("circular_shift(30, 0): %u\n", circular_shift(30, 0));
}

void
test_key_addition() {
    unsigned char out[16];
    unsigned char in1[] = {0, 2, 4, 6, 8, 10, 12, 14,
                           1, 3, 5, 7, 9, 11, 13, 15};
    unsigned char in2[] = {12, 14, 1, 3, 5, 7, 9, 11,
                           13, 15, 0, 2, 4, 6, 8, 10};
    key_addition(in1, in2, out);
    printf("key_addition(in1, in2, out): ");
    for (unsigned int i = 0; i < 16; i++)
        printf("%hhu ", out[i]);
    printf("\n");
}

void
test_inv_byte_substitution() {
    unsigned char out[16];
    unsigned char in[] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    inv_byte_substitution(in, out);
    printf("inv_byte_substitution(in, out): ");
    for (unsigned int i = 0; i < 16; i++)
        printf("%hhu ", out[i]);
    printf("\n");
}

void
test_nth_uniq_char() {
    char str1[] = "abcdefghijk";
    char str2[] = "aaabbbcccde";

    assert(nth_uniq_char(str1, 1) == 0);
    printf("nth_uniq_char(str1, 1) expected: %d\nactual: %d\n", 0,
           nth_uniq_char(str1, 1));
    assert(nth_uniq_char(str1, 2) == 1);
    assert(nth_uniq_char(str1, 3) == 2);
    assert(nth_uniq_char(str1, 4) == 3);
    printf("nth_uniq_char(str1, 4) expected: %d\nactual: %d\n", 3,
           nth_uniq_char(str1, 4));
    assert(nth_uniq_char(str1, 11) == 10);
    assert(nth_uniq_char(str1, 12) == 11);
    printf("nth_uniq_char(str1, 12) expected: %d\nactual: %d\n", 11,
           nth_uniq_char(str1, 12));
    assert(nth_uniq_char(str1, 13) == 11);
    printf("nth_uniq_char(str1, 13) expected: %d\nactual: %d\n", 11,
           nth_uniq_char(str1, 13));

    assert(nth_uniq_char(str2, 1) == 0);
    assert(nth_uniq_char(str2, 2) == 3);
    assert(nth_uniq_char(str2, 3) == 6);
    assert(nth_uniq_char(str2, 4) == 9);
    assert(nth_uniq_char(str2, 5) == 10);
    assert(nth_uniq_char(str2, 6) == 11);
    assert(nth_uniq_char(str2, 7) == 11);
}

void
test_max_unique_n_substr() {
    char str1[] = "xxxyziiijzz";
    char str2[] = "xxxyziiijzzy";
    char result[32] = {0};

    max_unique_n_substr(str1, result, 2);
    char exr2a[] = "xxxy";
    printf("max_unique_n_substr(str1, result, 2) expected: %s\nactual: %s\n",
           exr2a, result);

    memset(result, 0, 32);
    max_unique_n_substr(str1, result, 3);
    char exr3[] = "ziiijzz";
    assert(!strcmp(result, exr3));
    printf("max_unique_n_substr(str1, result, 3) expected: %s\nactual:%s\n",
           exr3, result);
    memset(result, 0, 32);
    max_unique_n_substr(str2, result, 3);
    assert(!strcmp(result, exr3));
    printf("max_unique_n_substr(str2, result, 3) expected: %s\nactual:%s\n",
           exr3, result);

    memset(result, 0, 32);
    max_unique_n_substr(str1, result, 4);
    char exr4[] = "xxxyziii";
    assert(!strcmp(result, exr4));
    printf("max_unique_n_substr(str1, result, 4) expected: %s\nactual:%s\n",
           exr4, result);
}

int
main(int argc, char **argv) {
    test_my_strlen();
    test_circular_shift();
    test_key_addition();
    test_inv_byte_substitution();
    test_nth_uniq_char();
    test_max_unique_n_substr();
    return 0;
}

void
my_strncpy(char *dest, char *src, unsigned int n) {
    unsigned int cpy_len = my_strlen(src) + 1;
    if (n < cpy_len) {
        cpy_len = n;
    }

    for (int i = 0; i < cpy_len; i++) {
        dest[i] = src[i];
    }
}

unsigned char inv_sbox[256] = {
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E,
    0x81, 0xF3, 0xD7, 0xFB, 0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87,
    0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB, 0x54, 0x7B, 0x94, 0x32,
    0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49,
    0x6D, 0x8B, 0xD1, 0x25, 0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92, 0x6C, 0x70, 0x48, 0x50,
    0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05,
    0xB8, 0xB3, 0x45, 0x06, 0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02,
    0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B, 0x3A, 0x91, 0x11, 0x41,
    0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8,
    0x1C, 0x75, 0xDF, 0x6E, 0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89,
    0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B, 0xFC, 0x56, 0x3E, 0x4B,
    0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59,
    0x27, 0x80, 0xEC, 0x5F, 0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D,
    0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF, 0xA0, 0xE0, 0x3B, 0x4D,
    0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63,
    0x55, 0x21, 0x0C, 0x7D};

// a code generator for the ALU chain in the 32-bit ALU
// see example_generator.cpp for inspiration

// make generator
// ./generator

#include <cstdio>
using std::printf;

int main() {
    int width = 32;
    for (int i = 0 ; i < width ; i ++) {
        printf("    alu1 a%d(out[%d], carryout[%d], A[%d], B[%d], carryout[%d], control);\n", i, i, i, i, i, i-1);
    }
    return 0;
}

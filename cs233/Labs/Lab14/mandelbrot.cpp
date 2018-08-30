#include "mandelbrot.h"
#include <xmmintrin.h>

// cubic_mandelbrot() takes an array of SIZE (x,y) coordinates --- these are
// actually complex numbers x + yi, but we can view them as points on a plane.
// It then executes 200 iterations of f, using the <x,y> point, and checks
// the magnitude of the result; if the magnitude is over 2.0, it assumes
// that the function will diverge to infinity.

// vectorize the code below using SIMD intrinsics
int *
cubic_mandelbrot_vector(float x[SIZE], float y[SIZE]) {
    static int ret[SIZE];
    float temp[4];
    //float x1, y1, x2, y2;
    __m128 acc, X, Y, X2, Y2, X_S, Y_S, MAG, x_i, y_i;

    MAG = _mm_set1_ps(M_MAG);
    MAG = _mm_mul_ps(MAG, MAG);

    for (int i = 0; i < SIZE; i += 4) {
        //x1 = y1 = 0.0;
        X = _mm_set1_ps(0.0);
        Y = _mm_set1_ps(0.0);
        x_i = _mm_loadu_ps(&x[i]);
        y_i = _mm_loadu_ps(&y[i]);

        acc = _mm_set1_ps(0.0);
        // Run M_ITER iterations
        for (int j = 0; j < M_ITER; j ++) {
            // Calculate x1^2 and y1^2
            // float x1_squared = x1 * x1;
            // float y1_squared = y1 * y1;
            X_S = _mm_mul_ps(X, X);
            Y_S = _mm_mul_ps(Y, Y);

            // Calculate the real piece of (x1 + (y1*i))^3 + (x + (y*i))
            // x2 = x1 * (x1_squared - 3 * y1_squared) + x[i];
            X2 = _mm_add_ps(_mm_mul_ps(X, _mm_sub_ps(X_S , _mm_mul_ps(_mm_set1_ps(3.0), Y_S))), x_i);

            // Calculate the imaginary portion of (x1 + (y1*i))^3 + (x + (y*i))
            // y2 = y1 * (3 * x1_squared - y1_squared) + y[i];
            Y2 = _mm_add_ps(_mm_mul_ps(Y, _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(3.0), X_S), Y_S)), y_i);

            // Use the resulting complex number as the input for the next
            // iteration

            X = X2;
            Y = Y2;
            // x1 = x2;
            // y1 = y2;
        }

        // caculate the magnitude of the result;
        // we could take the square root, but we instead just
        // compare squares
        X2 = _mm_mul_ps(X2, X2);
        Y2 = _mm_mul_ps(Y2, Y2);
        acc = _mm_add_ps(X2, Y2);
        acc = _mm_cmplt_ps(acc, MAG);
        _mm_storeu_ps(temp, acc);

        ret[i] = temp[0];
        ret[i+1] = temp[1];
        ret[i+2] = temp[2];
        ret[i+3] = temp[3];
        //ret[i] = ((x2 * x2) + (y2 * y2)) < (M_MAG * M_MAG);
    }

    return ret;
}

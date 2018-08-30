
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define OUT_SIZE 24
#define IN_SIZE 28
#define QUERY 0

#define SHITTY_UNROLL(i) \
do { \
    const DType* tmp_kernels = (DType*)&kernels[i][0]; \
    const DType temp = tmp_kernels[0] * tmp_unrolled[0] \
                     + tmp_kernels[1] * tmp_unrolled[1] \
                     + tmp_kernels[2] * tmp_unrolled[2] \
                     + tmp_kernels[3] * tmp_unrolled[3] \
                     + tmp_kernels[4] * tmp_unrolled[4] \
                     + tmp_kernels[10] * tmp_unrolled[10] \
                     + tmp_kernels[11] * tmp_unrolled[11] \
                     + tmp_kernels[12] * tmp_unrolled[12] \
                     + tmp_kernels[13] * tmp_unrolled[13] \
                     + tmp_kernels[14] * tmp_unrolled[14] \
                     + tmp_kernels[5] * tmp_unrolled[5] \
                     + tmp_kernels[6] * tmp_unrolled[6] \
                     + tmp_kernels[7] * tmp_unrolled[7] \
                     + tmp_kernels[8] * tmp_unrolled[8] \
                     + tmp_kernels[9] * tmp_unrolled[9] \
                     + tmp_kernels[15] * tmp_unrolled[15] \
                     + tmp_kernels[16] * tmp_unrolled[16] \
                     + tmp_kernels[17] * tmp_unrolled[17] \
                     + tmp_kernels[18] * tmp_unrolled[18] \
                     + tmp_kernels[19] * tmp_unrolled[19] \
                     + tmp_kernels[20] * tmp_unrolled[20] \
                     + tmp_kernels[21] * tmp_unrolled[21] \
                     + tmp_kernels[22] * tmp_unrolled[22] \
                     + tmp_kernels[23] * tmp_unrolled[23] \
                     + tmp_kernels[24] * tmp_unrolled[24]; \
    y[y_off] = temp; \
    y_off += y3; \
} while(0)

/*

*/

namespace mxnet
{
namespace op
{

__constant__ float kernels[50][25];

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    // max size of unrolled image, independent of kernel size
    // max((28 - K + 1) * K * 28)
    __shared__ DType unrolled[3360];//(29 - 14) * 14 * 28];
    const int y3 = OUT_SIZE * OUT_SIZE;
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    // unroll the input image
    // Note that my way of unroll is a little bit different from the one on the
    // slide. For example: a 4x4 image and 3x3 kernel
    // +---+---+---+---+
    // | 1 | 2 | 3 | 4 |
    // +---+---+---+---+
    // | 5 | 6 | 7 | 8 |
    // +---+---+---+---+
    // | 10| 11| 12| 13|
    // +---+---+---+---+
    // | 14| 15| 16| 17|
    // +---+---+---+---+
    // Will be unrolled to:
    // | 1 | 2 | 3 | 5 | 6 | 7 | 10 | 11 | 12 | 14 | 15 | 16 |...
    // | 2 | 3 | 4 | 6 | 7 | 8 | 11 | 12 | 13 | 15 | 16 | 17 |

    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    int row;
    int col;
    int offset = K * IN_SIZE - 1;
    for (; idx < IN_SIZE * IN_SIZE; idx += OUT_SIZE * OUT_SIZE) {
        row = idx / IN_SIZE;
        col = idx - row * IN_SIZE;
        int curr_idx = col + row * K;

        //               batch       channel  row  col
        DType curr = x4d(blockIdx.x, 0      , row, col);
        for (int i = 0; i < OUT_SIZE; ++i) {
            if (col - i < K && col - i >= 0) {
                unrolled[curr_idx] = curr;
            }
            curr_idx += offset;
        }
    }

    __syncthreads();

    row = threadIdx.y;
    col = threadIdx.x;
    int y_off = row * OUT_SIZE + col + M * OUT_SIZE * OUT_SIZE * blockIdx.x;
    DType* tmp_unrolled = &unrolled[(row + col * IN_SIZE) * K];

    SHITTY_UNROLL(0);
    SHITTY_UNROLL(1);
    SHITTY_UNROLL(2);
    SHITTY_UNROLL(3);
    SHITTY_UNROLL(4);
    SHITTY_UNROLL(5);
    SHITTY_UNROLL(6);
    SHITTY_UNROLL(7);
    SHITTY_UNROLL(8);
    SHITTY_UNROLL(9);
    SHITTY_UNROLL(10);
    SHITTY_UNROLL(11);
    SHITTY_UNROLL(12);
    SHITTY_UNROLL(13);
    SHITTY_UNROLL(14);
    SHITTY_UNROLL(15);
    SHITTY_UNROLL(16);
    SHITTY_UNROLL(17);
    SHITTY_UNROLL(18);
    SHITTY_UNROLL(19);
    SHITTY_UNROLL(20);
    SHITTY_UNROLL(21);
    SHITTY_UNROLL(22);
    SHITTY_UNROLL(23);
    SHITTY_UNROLL(24);
    SHITTY_UNROLL(25);
    SHITTY_UNROLL(26);
    SHITTY_UNROLL(27);
    SHITTY_UNROLL(28);
    SHITTY_UNROLL(29);
    SHITTY_UNROLL(30);
    SHITTY_UNROLL(31);
    SHITTY_UNROLL(32);
    SHITTY_UNROLL(33);
    SHITTY_UNROLL(34);
    SHITTY_UNROLL(35);
    SHITTY_UNROLL(36);
    SHITTY_UNROLL(37);
    SHITTY_UNROLL(38);
    SHITTY_UNROLL(39);
    SHITTY_UNROLL(40);
    SHITTY_UNROLL(41);
    SHITTY_UNROLL(42);
    SHITTY_UNROLL(43);
    SHITTY_UNROLL(44);
    SHITTY_UNROLL(45);
    SHITTY_UNROLL(46);
    SHITTY_UNROLL(47);
    SHITTY_UNROLL(48);
    SHITTY_UNROLL(49);

    #undef x4d
}

// Implemented with the assumption that there is only one level of
// convolution goes through this operator. And the input channel is
// always 1, and has shape 28 x 28

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    using namespace std;

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // batch size
    const int B = x.shape_[0];
    // output chanel
    const int M = y.shape_[1];
    // input chanel
    const int C = x.shape_[1];
    // height
    const int H = x.shape_[2];
    // width
    const int W = x.shape_[3];
    // we use square kernel, so this is the kernel width
    const int K = w.shape_[3];

    // Set the kernel dimensions
    dim3 gridDim(B, 1, 1);
    dim3 blockDim(OUT_SIZE, OUT_SIZE, 1);

    cudaMemcpyToSymbol(kernels, w.dptr_, M * K * K * C * sizeof(DType));

    // Call the kernel
    forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

#if QUERY
/* +++++++++++++++++++++++++++++++Device Query+++++++++++++++++++++++++++++++ */

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                cout << "No CUDA GPU has been detected" << endl;
                return;
            } else if (deviceCount == 1) {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
                //@@ message to be printed.
                cout << "There is 1 device supporting CUDA" << endl;
            } else {
                cout << "There are " << deviceCount << " devices supporting CUDA" << endl;
            }
        }

        cout << " Computational Capabilities: " << deviceProp.major << "."
             << deviceProp.minor << endl;
        cout << " Maximum global memory size: "
             << deviceProp.totalGlobalMem << endl;
        cout << " Maximum constant memory size: "
             << deviceProp.totalConstMem << endl;
        cout << " Maximum shared memory size per block: "
             << deviceProp.sharedMemPerBlock << endl;
        cout << " Maximum block dimensions: "
             << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1]
             << " x " << deviceProp.maxThreadsDim[2] << endl;
        cout << " Maximum grid dimensions: " << deviceProp.maxGridSize[0]
             << " x " << deviceProp.maxGridSize[1] << " x "
             << deviceProp.maxGridSize[2] << endl;
        cout << " Warp size: " << deviceProp.warpSize << endl;
    }
#endif

}


#undef OUT_SIZE
#undef IN_SIZE
#undef QUERY

}
}

#endif

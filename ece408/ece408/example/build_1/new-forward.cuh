#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IN_SIZE 28
#define IN_SIZE_SQ 784
#define OUT_SIZE 24
#define OUT_SIZE_SQ 576
#define MASK_SIZE 5
#define M_const 50
#define K_const 5

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__constant__ float Kc[M_const][MASK_SIZE][MASK_SIZE];
    
template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k) {
    // shared memory, oversided to minimze control divergence in loading and utilize burst
	__shared__ DType x_shared[OUT_SIZE_SQ * 2];
    #define y4d(i3,i2,i1,i0) y[(i3) * (M_const * OUT_SIZE_SQ) + (i2)*(OUT_SIZE_SQ) + (i1)*(OUT_SIZE) + i0]
    int b = blockIdx.x;
    int h = threadIdx.y;
    int w = threadIdx.x;
    int idx = h * OUT_SIZE + w;
    
    x_shared[idx] = x[b * IN_SIZE_SQ + idx];
    x_shared[idx + OUT_SIZE_SQ] = x[b * IN_SIZE_SQ + OUT_SIZE_SQ + idx];
    __syncthreads();
    
    DType result = 0;
    for (int m = 0; m < M_const; m++) {
        result = 0;
        for (int p = 0;  p < K_const; ++p) {
            for (int q = 0; q < K_const; ++q) {
                result += x_shared[(h + p) * IN_SIZE + w + q] * Kc[m][p][q];
            }
        }
        if (h < OUT_SIZE && w < OUT_SIZE) {
            y4d(b, m, h, w) = result;
        }
    }
    #undef y4d
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {

    // Use mxnet's CHECK_EQ to do assertions.
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    //cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0];
    
    cudaMemcpyToSymbol(Kc, w.dptr_, M_const * MASK_SIZE * MASK_SIZE * sizeof(DType));
    
    dim3 blockDim(OUT_SIZE, OUT_SIZE, 1);
    dim3 gridDim(B, 1, 1);
    forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif

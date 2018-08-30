#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 24
#define IN_SIZE 28
#define OUT_SIZE 24
#define MASK_SIZE 5
#define M_const 50
#define C_const 1
#define K_const 5

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__constant__ float Kc[M_const][C_const][MASK_SIZE][MASK_SIZE];
    
template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k) {

	__shared__ DType x_shared[BLOCK_SIZE * BLOCK_SIZE * 2];

    #define y4d(i3,i2,i1,i0) y[(i3) * (M_const * OUT_SIZE * OUT_SIZE) + (i2)*(OUT_SIZE * OUT_SIZE) + (i1)*(OUT_SIZE) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (K_const * K_const) + (i2)*(K_const * K_const) + (i1)*(K_const) + i0]

    int h = threadIdx.y;
    int w = threadIdx.x;
    
    int idx = h * BLOCK_SIZE + w;
    x_shared[idx] = x[blockIdx.x * IN_SIZE * IN_SIZE + idx];
    x_shared[idx + BLOCK_SIZE * BLOCK_SIZE] = x[blockIdx.x * IN_SIZE * IN_SIZE + BLOCK_SIZE * BLOCK_SIZE + idx];
    __syncthreads();
    
    DType result = 0;
    for (int m = 0; m < M_const; m++) {
        result = 0;
        for (int p = 0;  p < K_const; ++p) {
            for (int q = 0; q < K_const; ++q) {
                result += x_shared[(threadIdx.y + p) * IN_SIZE + threadIdx.x + q] * Kc[m][0][p][q];
            }
        }
        /*
        if (h < OUT_SIZE && w < OUT_SIZE) {
            y4d(blockIdx.x, m, h, w) = result;
        }
         */
        y4d(blockIdx.x, m, h, w) = result;
    }
    
    #undef y4d
    #undef k4d
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
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    
    //size_t shmem_size = sizeof(DType) * ((BLOCK_SIZE + K - 1) * (BLOCK_SIZE + K - 1) + K * K);
    
    //DType k_host[M_const * C_const * MASK_SIZE * MASK_SIZE];
    //cudaMemcpy(k_host, w.dptr_, M_const * C_const * MASK_SIZE * MASK_SIZE * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(Kc, w.dptr_, M_const * C_const * MASK_SIZE * MASK_SIZE * sizeof(DType));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(B, 1, 1);
    forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

    // Call the kernel
    // forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 24
#define MASK_SIZE 5
#define M_const 50
#define C_const 1
#define M_SIZE 50

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__constant__ float Kc[M_const][C_const][MASK_SIZE][MASK_SIZE];
    
template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

	int x_dim = BLOCK_SIZE + K - 1;
	__shared__ DType x_shared[(BLOCK_SIZE + MASK_SIZE - 1) * (BLOCK_SIZE + MASK_SIZE - 1)];
	//__shared__ DType w_shared[64];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    

    int b = blockIdx.x;
    int m_begin = blockIdx.y * M_SIZE;
    int W_grid = ceil(W_out / 1.0 / BLOCK_SIZE);
    int h_start = blockIdx.z / W_grid * BLOCK_SIZE;
    int h = h_start + threadIdx.y;
    int w_start = (blockIdx.z % W_grid) * BLOCK_SIZE;
    int w = w_start + threadIdx.x;
    
    for (int m = m_begin; m < m_begin + M_SIZE; m++) {
        DType result = 0;
        for (int c = 0; c < C; c++) {
            for (int i = h; i < h_start + x_dim; i += BLOCK_SIZE) {
                for (int j = w; j < w_start + x_dim; j += BLOCK_SIZE) {
                    if (i < H && j < W) {
                        x_shared[(i - h_start) * x_dim + j - w_start] = x4d(b, c, i, j);
                    } else {
                        x_shared[(i - h_start) * x_dim + j - w_start] = 0;
                    }
                }
            }

            __syncthreads();

            for (int p = 0;  p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    //result += x_shared[(threadIdx.y + p) * x_dim + threadIdx.x + q] * w_shared[p * K + q];
                    result += x_shared[(threadIdx.y + p) * x_dim + threadIdx.x + q] * Kc[m][c][p][q];
                }
            }
        }
        if (h < H_out && w < W_out) {
            y4d(b, m, h, w) = result;
        }
    }
    

    #undef y4d
    #undef x4d
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
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    int zDim = ceil(H_out / 1.0 / BLOCK_SIZE) * ceil( W_out / 1.0 / BLOCK_SIZE);
    //size_t shmem_size = sizeof(DType) * ((BLOCK_SIZE + K - 1) * (BLOCK_SIZE + K - 1) + K * K);
    
    //DType k_host[M_const * C_const * MASK_SIZE * MASK_SIZE];
    //cudaMemcpy(k_host, w.dptr_, M_const * C_const * MASK_SIZE * MASK_SIZE * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(Kc, w.dptr_, M_const * C_const * MASK_SIZE * MASK_SIZE * sizeof(DType));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(B, M / M_SIZE, zDim);
    forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // Call the kernel
    // forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif

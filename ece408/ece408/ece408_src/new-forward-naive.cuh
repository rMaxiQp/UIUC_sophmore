
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 16
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.
We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
*/

//naive implementation
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  int H_out = H - K + 1;
  int W_out = W - K + 1;

  int n, m, h, w;
  n = blockIdx.x;
  m = blockIdx.y;
  h = blockDim.y * (blockIdx.z / ((W_out - 1) / BLOCK_SIZE + 1)) + threadIdx.y;
  w = blockDim.x * (blockIdx.z % ((W_out - 1) / BLOCK_SIZE + 1)) + threadIdx.x;
  if(h < H_out && w < W_out) {
    int c, p, q;
    float acc = 0.0;
    for(c = 0; c < C; c++) {
      for(p = 0; p < K; p++) {
        for(q = 0; q < K; q++) {
          acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
        }
      }
    }
    y4d(n, m, h, w) = acc;
  }
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
  // Extract the tensor dimensions into B,M,C,H,W,K
  int B = x.shape_[0]; //batch
  int M = y.shape_[1]; //output num
  int C = x.shape_[1]; //input num
  int H = x.shape_[2]; //input height
  int W = x.shape_[3]; //input width
  int K = w.shape_[3]; //mask height && width

  int Z = ((H - K) / BLOCK_SIZE + 1) * ((W - K) / BLOCK_SIZE + 1); //H_grid * W_grid

  // Set the kernel dimensions
  dim3 gridDim(B, M, Z);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Call the kernel
  forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#undef y4d
#undef x4d
#undef k4d
#endif

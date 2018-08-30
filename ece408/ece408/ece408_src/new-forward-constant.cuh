
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdlib.h>
#include <stdio.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 16
#define K 5 //mask height && width (know from fprintf)
#define M_first 6 //output num for forward
#define M_second 16 //output num for backward
#define H_first 64 //input height for forward
#define W_first 64 //input width for forward
#define H_second 30 //input height for backward
#define W_second 30 //input width for backward
#define C_first 1 //input num for forward
#define C_second 6 //input num for backward
/*
 W_out = W - K + 1
 H_out = H - K + 1
 */
#define W_out_first 60 //output width for forward
#define H_out_first 60 //output height for forward
#define W_out_second 26 //output width for backward
#define H_out_second 26 //output height for backward

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d_first(i3, i2, i1, i0) y[(i3) * (M_first * H_out_first * W_out_first) + (i2) * (H_out_first * W_out_first) + (i1) * (W_out_first) + i0]
#define y4d_second(i3, i2, i1, i0) y[(i3) * (M_second * H_out_second * W_out_second) + (i2) * (H_out_second * W_out_second) + (i1) * (W_out_second) + i0]
#define x4d_first(i3, i2, i1, i0) x[(i3) * (C_first * H_first * W_first) + (i2) * (H_first * W_first) + (i1) * (W_first) + i0]
#define x4d_second(i3, i2, i1, i0) x[(i3) * (C_second * H_second * W_second) + (i2) * (H_second * W_second) + (i1) * (W_second) + i0]
#define k4d_first(arr, i3, i2, i1, i0) arr[(i3) * (C_first * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k4d_second(arr, i3, i2, i1, i0) arr[(i3) * (C_second * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.
We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
*/

__constant__ float Mask[2400];

__global__ void first_forward_kernel(float *y, const float *x, const int B)
{
  int n, m, h, w;
  n = blockIdx.x;
  m = blockIdx.y;
  h = blockDim.y * (blockIdx.z / ((W_out_first - 1) / BLOCK_SIZE + 1)) + threadIdx.y;
  w = blockDim.x * (blockIdx.z % ((W_out_first - 1) / BLOCK_SIZE + 1)) + threadIdx.x;
  if(h < H_out_first && w < W_out_first)
  {
    int p;
    float acc = 0.0;
    for(p = 0; p < K; p++)
    {
      acc += x4d_first(n, 0, h + p, w + 0) * k4d_first(Mask, m, 0, p, 0);
      acc += x4d_first(n, 0, h + p, w + 1) * k4d_first(Mask, m, 0, p, 1);
      acc += x4d_first(n, 0, h + p, w + 2) * k4d_first(Mask, m, 0, p, 2);
      acc += x4d_first(n, 0, h + p, w + 3) * k4d_first(Mask, m, 0, p, 3);
      acc += x4d_first(n, 0, h + p, w + 4) * k4d_first(Mask, m, 0, p, 4);
    }
    y4d_first(n, m, h, w) = acc;
  }
}

__global__ void second_forward_kernel(float *y, const float *x, const float *k, const int B)
{
  int n, m, h, w;
  n = blockIdx.x;
  m = blockIdx.y;
  h = blockDim.y * (blockIdx.z / ((W_out_second - 1) / BLOCK_SIZE + 1)) + threadIdx.y;
  w = blockDim.x * (blockIdx.z % ((W_out_second - 1) / BLOCK_SIZE + 1)) + threadIdx.x;
  if(h < H_out_second && w < W_out_second)
  {
    int c, p;
    float acc = 0.0;
    for(c = 0; c < C_second; c++)
    {
      for(p = 0; p < K; p++)
      {
        acc += x4d_second(n, c, h + p, w + 0) * k4d_second(Mask, m, c, p, 0);
        acc += x4d_second(n, c, h + p, w + 1) * k4d_second(Mask, m, c, p, 1);
        acc += x4d_second(n, c, h + p, w + 2) * k4d_second(Mask, m, c, p, 2);
        acc += x4d_second(n, c, h + p, w + 3) * k4d_second(Mask, m, c, p, 3);
        acc += x4d_second(n, c, h + p, w + 4) * k4d_second(Mask, m, c, p, 4);
      }
    }
    y4d_second(n, m, h, w) = acc;
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
  int M = y.shape_[1]; //output num {6 when forward, 16 when backward}

  if(M_first == M) //first
  {
    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * M_first * C_first * K * K);

    // Set the kernel dimensions
    dim3 gridDim_forward(B, M_first, 16); //((H_first - K) / BLOCK_SIZE + 1) * ((W_first - K) / BLOCK_SIZE + 1)   [H_grid * W_grid]
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Call the kernel
    first_forward_kernel<<<gridDim_forward, blockDim>>>(y.dptr_,x.dptr_, B);
  }
  else //second
  {
    // 16 * 6 * 5 * 5 = 2400
    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * M_second * C_second * K * K);

    // Set the kernel dimensions
    dim3 gridDim_second(B, M_second, 4);//((H_second - K) / BLOCK_SIZE + 1) * ((W_second - K) / BLOCK_SIZE + 1) [H_grid * W_grid]
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Call the kernel
    second_forward_kernel<<<gridDim_second, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B);
  }

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

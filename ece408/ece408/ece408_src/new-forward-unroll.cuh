
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

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
  #define W_unroll_first 3600 //W_out_first * H_out_first
  #define H_unroll_first 25 //C_first * K * K
  #define W_unroll_second 676 //W_out_second * H_out_second
  #define H_unroll_second 150 //C_second * K * K
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
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define MIN(a, b) ((a < b) ? a : b)
  #define MAX(a, b) ((a > b) ? a : b)

/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.
We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
*/

__constant__ float Mask[2400];

__global__ void forward_kernel(float *y, const float *x)
{
  int n, m, h, w;
  n = blockIdx.x;
  m = blockIdx.y;
  //((W_out_second - 1) / BLOCK_SIZE + 1))
  h = blockDim.y * (blockIdx.z / 2) + threadIdx.y;
  w = blockDim.x * (blockIdx.z % 2) + threadIdx.x;
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

__global__ void unroll_first(float* x, float* X_unroll)
{
  int c, s, h_out, w_out, h_unroll, w_base, p, q;
  int t = blockIdx.x * 1024 + threadIdx.x;

  if (t < W_unroll_first) {
    c = t / W_unroll_first;
    s = t % W_unroll_first;
    h_out = s / W_out_first;
    w_out = s % W_out_first;
    h_unroll = h_out * W_out_first + w_out;
    w_base = c * K * K;
    for(p = 0; p < K; p++) {
      for(q = 0; q < K; q++) {
        X_unroll[(w_base + p * K + q) * W_unroll_first + h_unroll] = x4d_first(0, c, h_out + p, w_out + q);
      }
    }
  }
}

__global__ void matmat_first(float *input, float *output)
{

  __shared__ float xMem[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float k_Mem[BLOCK_SIZE][BLOCK_SIZE];

  int y = threadIdx.y;
  int x = threadIdx.x;
  int Row = blockIdx.y * blockDim.y + y;
  int Col = blockIdx.x * blockDim.x + x;

  float Pvalue = 0.0;
  for (int m = 0; m < (BLOCK_SIZE + H_unroll_first - 1)/BLOCK_SIZE; ++m)
  {
    // Collaborative loading of M and N tiles into shared memory
    if(Row < M_first && m * BLOCK_SIZE + x < H_unroll_first)
      k_Mem[y][x] = Mask[Row * H_unroll_first + m * BLOCK_SIZE + x];
    else
      k_Mem[y][x] = 0.0;

     if(m * BLOCK_SIZE + y < H_unroll_first && Col < W_unroll_first)
       xMem[y][x] = input[(m*BLOCK_SIZE + y) * W_unroll_first + Col];
     else
       xMem[y][x] = 0.0;

     __syncthreads();
     for (int k = 0; k < BLOCK_SIZE; ++k)
     {
       Pvalue += k_Mem[y][k] * xMem[k][x];
     }
      __syncthreads();
   }

   if(Row < M_first && Col < W_unroll_first)
   {
     output[W_unroll_first * Row + Col] = Pvalue;
   }
 }

 __global__ void unroll_second(float* X, float* X_unroll)
 {
   int c, s, h_out, w_out, h_unroll, w_base, p, q;
   int t = blockIdx.x * 1024 + threadIdx.x;

   if (t < C_second * W_unroll_second) {
     c = t / W_unroll_second;
     s = t % W_unroll_second;
     h_out = s / W_out_second;
     w_out = s % W_out_second;
     h_unroll = h_out * W_out_second + w_out;
     w_base = c * K * K;
     for(p = 0; p < K; p++) {
         X_unroll[(w_base + p * K + 0) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 0];
         X_unroll[(w_base + p * K + 1) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 1];
         X_unroll[(w_base + p * K + 2) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 2];
         X_unroll[(w_base + p * K + 3) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 3];
         X_unroll[(w_base + p * K + 4) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 4];
     }
   }
 }

 __global__ void matmat_second(float *input, float *output)
 {

   __shared__ float xMem[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ float k_Mem[BLOCK_SIZE][BLOCK_SIZE];

   int y = threadIdx.y;
   int x = threadIdx.x;
   int Row = blockIdx.y * blockDim.y + y;
   int Col = blockIdx.x * blockDim.x + x;

   float Pvalue = 0.0;
   for (int m = 0; m < (BLOCK_SIZE + H_unroll_second - 1)/BLOCK_SIZE; ++m)
   {
     // Collaborative loading of M and N tiles into shared memory
     if(Row < M_second && m * BLOCK_SIZE + x < H_unroll_second)
       k_Mem[y][x] = Mask[Row * H_unroll_second + m * BLOCK_SIZE + x];
     else
       k_Mem[y][x] = 0.0;

      if(m * BLOCK_SIZE + y < H_unroll_second && Col < W_unroll_second)
        xMem[y][x] = input[(m * BLOCK_SIZE + y) * W_unroll_second + Col];
      else
        xMem[y][x] = 0.0;

      __syncthreads();
      for (int k = 0; k < BLOCK_SIZE; ++k)
      {
        Pvalue += k_Mem[y][k] * xMem[k][x];
      }
       __syncthreads();
    }

    if(Row < M_second && Col < W_unroll_second)
    {
      output[W_unroll_second * Row + Col] = Pvalue;
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

  // Call the kernel
  if(M == M_first)
  {
    float *device_input;
    cudaMalloc((void **)&device_input, K * K * M * H_out_first * W_out_first * sizeof(float));
    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * 150);

    int num_blocks = (H_out_first * W_out_first - 1) / 1024 + 1;

    dim3 dimGrid((W_out_first * H_out_first- 1)/BLOCK_SIZE + 1, (M_first - 1) /BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    for(int i = 0; i < B; i++)
    {
      unroll_first<<<num_blocks, 1024>>>(x.dptr_ + C_first * i * W_first * H_first, device_input);
      matmat_first<<<dimGrid, dimBlock>>>(device_input, y.dptr_ + M * i * W_out_first * H_out_first);
    }

    cudaFree(device_input);
  }
  else
  {
    float *device_input;
    cudaMalloc((void **)&device_input, C_second * K * K * M * H_out_first * W_out_first * sizeof(float));
    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * 2400);

    int num_blocks = (C_second * W_out_second * H_out_second - 1) / 1024 + 1;

    dim3 dimGrid((W_out_second * H_out_second- 1)/BLOCK_SIZE + 1, (M_second - 1) /BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    for(int i = 0; i < B; i++)
    {
      unroll_second<<<num_blocks, 1024>>>(x.dptr_ + C_second * i * W_second * H_second, device_input);
      matmat_second<<<dimGrid, dimBlock>>>(device_input, y.dptr_ + M_second * i * W_out_second * H_out_second);
    }

    cudaFree(device_input);
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

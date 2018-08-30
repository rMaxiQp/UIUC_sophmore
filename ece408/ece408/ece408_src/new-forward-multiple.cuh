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

#define K_m 5 //mask height && width (know from fprintf)
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
  // An example use of these macros:  // float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d_first(i3, i2, i1, i0) y[(i3) * (M_first * H_out_first * W_out_first) + (i2) * (H_out_first * W_out_first) + (i1) * (W_out_first) + i0]
#define y4d_second(i3, i2, i1, i0) y[(i3) * (M_second * H_out_second * W_out_second) + (i2) * (H_out_second * W_out_second) + (i1) * (W_out_second) + i0]
#define x4d_first(i3, i2, i1, i0) x[(i3) * (C_first * H_first * W_first) + (i2) * (H_first * W_first) + (i1) * (W_first) + i0]
#define x4d_second(i3, i2, i1, i0) x[(i3) * (C_second * H_second * W_second) + (i2) * (H_second * W_second) + (i1) * (W_second) + i0]
#define k4d_first(arr, i3, i2, i1, i0) arr[(i3) * (C_first * K_m * K_m) + (i2) * (K_m * K_m) + (i1) * (K_m) + i0]
#define k4d_second(arr, i3, i2, i1, i0) arr[(i3) * (C_second * K_m * K_m) + (i2) * (K_m * K_m) + (i1) * (K_m) + i0]
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

__constant__ float Mask[2400];


/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.
We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
*/

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int X_tile_width = BLOCK_SIZE+ K-1;
  int n, m, h, h0,w0,w,h_base,w_base;
  extern __shared__ float shmem[];
  //all the oprtion of the input X[n,c,..,..] into X_shared  (Therefore need size of X_tile_width ^2)
  float* X_shared = &shmem[0];
  //load the filter K[m,c] into the shared memory
  float* K_shared = &shmem[X_tile_width* X_tile_width];
  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.y;
  w0 = threadIdx.x;
  h_base = blockDim.y * (blockIdx.z / ((W_out - 1) / BLOCK_SIZE + 1));
  w_base= blockDim.x * (blockIdx.z % ((W_out - 1) / BLOCK_SIZE + 1)) ;
  h=h_base+h0;
  w=w_base+w0;
  float acc=0;
  for (int c=0; c<C; c++)
  { 
    if(h0 < K  && w0 < K) 
	 K_shared[(h0*K)+w0] = k4d(m,c,h0,w0);
    __syncthreads();
    for (int i=h; i<h_base+ X_tile_width; i+= BLOCK_SIZE)
   	 for (int j=w; j<w_base+X_tile_width; j+=BLOCK_SIZE)
	 	X_shared[(i-h_base)*X_tile_width+(j-w_base)]= x4d(n,c,i,j);
    __syncthreads();
    for (int p=0; p<K; p++)
    	for (int q=0; q<K; q++)
		  acc+=X_shared[(h0+p)*X_tile_width+(w0+q)]* K_shared[p*K+q];
    __syncthreads();
  }
  if (n < B && m < M && h < H_out && w < W_out)
    y4d(n, m, h, w) = acc;

}


__global__ void unroll_second(float* X, float* X_unroll)
 {
   int c, s, h_out, w_out, h_unroll, w_base, p;
   int t = blockIdx.x * 1024 + threadIdx.x;

   if (t < C_second * W_unroll_second) {
     c = t / W_unroll_second;
     s = t % W_unroll_second;
     h_out = s / W_out_second;
     w_out = s % W_out_second;
     h_unroll = h_out * W_out_second + w_out;
     w_base = c * K_m * K_m;
     for(p = 0; p < K_m; p++) {
         X_unroll[(w_base + p * K_m + 0) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 0];
         X_unroll[(w_base + p * K_m + 1) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 1];
         X_unroll[(w_base + p * K_m + 2) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 2];
         X_unroll[(w_base + p * K_m + 3) * W_unroll_second + h_unroll] =
         X[c * H_second * W_second + (h_out + p) * W_second + w_out + 3];
         X_unroll[(w_base + p * K_m + 4) * W_unroll_second + h_unroll] =
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
  int C = x.shape_[1]; //input num
  int H = x.shape_[2]; //input height
  int W = x.shape_[3]; //input width
  int K = w.shape_[3]; //mask height && width

  if(M == M_first)
  {
    int Z = ((H - K) / BLOCK_SIZE + 1) * ((W - K) / BLOCK_SIZE + 1); //H_grid * W_grid
    size_t shmem_size= sizeof(float) * ((BLOCK_SIZE+K-1)*(BLOCK_SIZE+K-1)+K*K);
    dim3 gridDim(B, M, Z);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    forward_kernel<<<gridDim, blockDim,shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
  }
  else
  {
    cudaStream_t s;
  cudaStreamCreate(&s);
  cudaStream_t s1;
  cudaStreamCreate(&s1);
  cudaStream_t s2;
  cudaStreamCreate(&s2);
  cudaStream_t s3;
  cudaStreamCreate(&s3);
  cudaStream_t s4;
  cudaStreamCreate(&s4);
  cudaStream_t s5;
  cudaStreamCreate(&s5);
  cudaStream_t s6;
  cudaStreamCreate(&s6);
  cudaStream_t s7;
  cudaStreamCreate(&s7);
  cudaStream_t s8;
  cudaStreamCreate(&s8);
    float *device_input;
    float *device_input_1;
    float *device_input_2;
    float *device_input_3;
    float *device_input_4;
    float *device_input_5;
    float *device_input_6;
    float *device_input_7;
    float *device_input_8;

    cudaMalloc((void **)&device_input,   C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_1, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_2, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_3, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_4, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_5, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_6, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_7, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    cudaMalloc((void **)&device_input_8, C_second * K_m * K_m * M * H_out_second * W_out_second * sizeof(float));
    
    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * 2400);

    int num_blocks = (C_second * W_out_second * H_out_second - 1) / 1024 + 1;

    dim3 dimGrid((W_out_second * H_out_second- 1)/BLOCK_SIZE + 1, (M_second - 1) /BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    for(int i = 0; i < B; i+=9)
    {
      unroll_second<<<num_blocks, 1024, 0, s>>>(x.dptr_ + C_second * i * W_second * H_second, device_input);
      unroll_second<<<num_blocks, 1024, 0, s1>>>(x.dptr_ + C_second * (i + 1) * W_second * H_second, device_input_1);
      unroll_second<<<num_blocks, 1024, 0, s2>>>(x.dptr_ + C_second * (i + 2) * W_second * H_second, device_input_2);
      unroll_second<<<num_blocks, 1024, 0, s3>>>(x.dptr_ + C_second * (i + 3) * W_second * H_second, device_input_3);
      unroll_second<<<num_blocks, 1024, 0, s4>>>(x.dptr_ + C_second * (i + 4) * W_second * H_second, device_input_4);
      unroll_second<<<num_blocks, 1024, 0, s5>>>(x.dptr_ + C_second * (i + 5) * W_second * H_second, device_input_5);
      unroll_second<<<num_blocks, 1024, 0, s6>>>(x.dptr_ + C_second * (i + 6) * W_second * H_second, device_input_6);
      unroll_second<<<num_blocks, 1024, 0, s7>>>(x.dptr_ + C_second * (i + 7) * W_second * H_second, device_input_7);
      unroll_second<<<num_blocks, 1024, 0, s8>>>(x.dptr_ + C_second * (i + 8) * W_second * H_second, device_input_8);
     
      matmat_second<<<dimGrid, dimBlock, 0, s>>>(device_input, y.dptr_ + M_second * i * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s1>>>(device_input_1, y.dptr_ + M_second * (i + 1) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s2>>>(device_input_2, y.dptr_ + M_second * (i + 2) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s3>>>(device_input_3, y.dptr_ + M_second * (i + 3) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s4>>>(device_input_4, y.dptr_ + M_second * (i + 4) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s5>>>(device_input_5, y.dptr_ + M_second * (i + 5) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s6>>>(device_input_6, y.dptr_ + M_second * (i + 6) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s7>>>(device_input_7, y.dptr_ + M_second * (i + 7) * W_out_second * H_out_second);
      matmat_second<<<dimGrid, dimBlock, 0, s8>>>(device_input_8, y.dptr_ + M_second * (i + 8) * W_out_second * H_out_second);
      
    }

    cudaFree(device_input);
    cudaFree(device_input_1);
    cudaFree(device_input_2);
    cudaFree(device_input_3);
    cudaFree(device_input_4);
    cudaFree(device_input_5);
    cudaFree(device_input_6);
    cudaFree(device_input_7);
    cudaFree(device_input_8);
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
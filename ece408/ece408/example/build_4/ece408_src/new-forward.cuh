#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 28
#define NUM_OF_PIC 2

namespace mxnet
{
namespace op
{

__constant__ float constant_kernel[1250];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    extern __shared__ float x_shared[]; 

    int block_num = blockIdx.x; 
    int local_h = threadIdx.y; 
    int local_w = threadIdx.x; 
    int input_size = 28 * 28; 
    int output_size = 24 * 24; 
    int x_base = block_num * NUM_OF_PIC * 28 * 28 + local_h * 28 + local_w; 
    int y_base = 0; 

    float sum = 0.0; 

    int i, j;
    
    // load weights for W [m, c,..],
    // h0 and w0 used as shorthand for threadIdx.x
    // and threadIdx.y
    // load tile from X[n, c,…] into shared memory
    for (int index = 0; index < NUM_OF_PIC; index++) {
        x_shared[index * input_size + local_h * 28 + local_w] = x[x_base]; 
        x_base += input_size; //threads in each block are as adjacent as images
    }

    __syncthreads(); 

    if (local_h < 24 && local_w < 24) {
        y_base = block_num * NUM_OF_PIC * 50 * output_size + local_h * 24 + local_w; 
        for (int index = 0; index < NUM_OF_PIC; index++) {
            for (int m = 0; m < 50; m++) {
                sum = 0.0; 
                for (i = 0; i < 5; i++) {
                    for (j = 0; j < 5; j++) {
                        sum += x_shared[index * input_size + (local_h + i) * 28 + local_w + j] * 
                            constant_kernel[m * 25 + i * 5 + j]; 
                    }
                }
                y[y_base + m * output_size] = sum; 
            }
            y_base += 50 * output_size; 
        }
    }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    // Set the kernel dimensions
    dim3 gridDim(B / 2, 1, 1); // too many blocks? no way. only 28x28 image
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // allocate constant memory
    cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float) * 1250, 0, cudaMemcpyDeviceToDevice);

    // Call the kernel
    size_t shmem_size = sizeof(float) * ( 2 * (TILE_WIDTH) * (TILE_WIDTH) );
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K); // legal? 

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif

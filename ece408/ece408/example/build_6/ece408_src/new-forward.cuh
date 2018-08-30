/* This code only works with input data is 28x28
 * and assumens inputs only has 1 channel，kernel is 5x5
 * and we have 50 output channel
 */

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iostream>
#include "./forward-kernel-macro.cuh"

using namespace std;

namespace mxnet
{
namespace op
{

#define TILE_WIDTH      24
#define INPUT_WIDTH     28
#define KERNEL_WIDTH    5
#define KERNEL_SIZE     25
#define H_OUT           24
#define W_OUT           24
#define OUT_SIZE        576 //24x24
#define IN_SIZE         784 //28x28
#define OUT_CHANNEL     50

// constant memory
__device__ __constant__ float MASK[OUT_CHANNEL*KERNEL_SIZE];

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y1, const DType *x1, DType *y2, const DType *x2) {
    // tx - x in input, ty - y in input, bx - batch
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x;// by = blockIdx.y;

    // shared memory to cache input for convolution
    // first image & second image
    DType *y_curr = y1;
    __shared__ DType T[IN_SIZE*2];
    int idx = ty*TILE_WIDTH+tx;
    int xidx = bx * IN_SIZE + idx;
    T[idx] = x1[xidx];
    T[idx+IN_SIZE] = x2[xidx];
    idx += OUT_SIZE;
    xidx += OUT_SIZE;
    if(idx<IN_SIZE){
        T[idx] = x1[xidx];
        T[idx+IN_SIZE] = x2[xidx];
    }
    __syncthreads();
    // see forward-kernel-macro.cuh for more details.
    // DType local_array[25]; // load tiled shared memory into local array which is register.
    DType t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24;
    const int yidx = bx * OUT_CHANNEL * OUT_SIZE + ty * W_OUT + tx;
    int y_idx = yidx;
    int a0 = ty * INPUT_WIDTH + tx;//, a1 = a0 + INPUT_WIDTH, a2 = a1 + INPUT_WIDTH;
    // int a3 = a2 + INPUT_WIDTH, a4 = a3 + INPUT_WIDTH;
    DType *T0 = &T[a0];// *T1 = T0 + INPUT_WIDTH, *T2 = T1 + INPUT_WIDTH, *T3 = T2 + INPUT_WIDTH, *T4 = T2 + INPUT_WIDTH;
    int mm = 0;
    _LOAD_LOCAL_ARRAY_;
    _SHITTY_UNROLL_;
    // for (int i = 0; i<50; ++i){
    //     _SHITTY_CONV_;
    // }
    // second image
    y_curr = y2;
    // see forward-kernel-macro.cuh for more details.
    y_idx = yidx;
    T0 = &T[a0+IN_SIZE]; 
    //T1 = &T[a0], *T1 = T0 + INPUT_WIDTH, *T2 = T1 + INPUT_WIDTH, *T3 = T2 + INPUT_WIDTH, *T4 = T2 + INPUT_WIDTH;
    mm = 0; 
    _LOAD_LOCAL_ARRAY_;
    _SHITTY_UNROLL_;
    // for (int i = 0; i<50; ++i){
    //     _SHITTY_CONV_;
    // }
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {
    // makes the run log clearer
    // cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    // batch size
    const int B = x.shape_[0];
    // output chanel
    const int M = y.shape_[1];
    // input chanel
    const int C = x.shape_[1];
    // input height
    const int H = x.shape_[2];
    // input width
    const int W = x.shape_[3];
    // we use square kernel, so this is the kernel width
    const int K = k.shape_[3];
    // output Height and weight
    // const int OH = H - K + 1;
    // const int OW = W - K + 1;

    // log the input/output dimensions
    // cout<<"batch: "<<B<<", output channel: "<<M<<", input channel: "<<C<<", input size: "<<H<<"x"<<W<<endl;
    // cout<<"kernel width: "<<K<<endl;
    // cout<<"input size: "<<y.shape_[2]<<"x"<<y.shape_[3]<<endl;

    // put kernel into constant memory
    cudaMemcpyToSymbol(MASK, k.dptr_, sizeof(DType) * C * M * K * K);

    // Set the kernel dimensions
    dim3 gridDim(B/4, 1, 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    // forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    for(int ii = 0; ii<B/2; ii += B/4)
        forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_+ii*OUT_CHANNEL*OUT_SIZE, x.dptr_+ii*IN_SIZE, 
                                                          y.dptr_+(B/2+ii)*OUT_CHANNEL*OUT_SIZE, x.dptr_+(B/2+ii)*IN_SIZE);
    //forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_+5000*50*24*24, x.dptr_+5000*28*28, k.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to do device sync.
    cudaDeviceSynchronize();

    // cout<<"---------------------------------------------------------"<<endl;
}

}
}
#endif
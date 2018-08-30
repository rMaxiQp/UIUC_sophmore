#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 24
#define B 10000
#define batch_size 2000
#define M 50
#define C 1
#define H 28
#define W 28
#define K 5
#define H_out 24
#define W_out 24
#define filter_size 25
#define input_size 784
#define output_size 576
#define TOTAL 28800

#include <mxnet/base.h>
namespace mxnet
{
namespace op
{
__constant__ float filters[1250];
texture<float, 1, cudaReadModeElementType> images;
template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y) {
    float acc = 0;
    for(int b=blockIdx.x; b<10000; b+=batch_size){
        for(int m=0; m<50; m++){
            for(int j=0; j<K; j++)
                for(int i=0; i<K; i++)
                    acc+=tex1Dfetch(images, b*input_size + (threadIdx.y+j)*28 + threadIdx.x+i)*filters[m*25 + j*5 +i];
            y[b*TOTAL + m*output_size + threadIdx.y*W_out + threadIdx.x] = acc;
            acc = 0;
        }
    }
}
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {
    cudaBindTexture(0, images , x.dptr_, sizeof(float) * B * input_size);
    cudaMemcpyToSymbol(filters, k.dptr_, sizeof(float) * 1250);
    forward_kernel<gpu, DType><<<batch_size, dim3(TILE_WIDTH, TILE_WIDTH, 1), 0, y.stream_->stream_>>>(y.dptr_);
    cudaUnbindTexture(images);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}
}
}
#endif

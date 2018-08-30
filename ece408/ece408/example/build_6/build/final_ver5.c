#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 24
#define B 10000
#define batch_size 2500
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
__global__ void forward_kernel(DType *op) {
    int x = threadIdx.x, y = threadIdx.y;
    for(int b=blockIdx.x; b<10000; b+=batch_size){
        for(int m=0; m<50; m++){
            op[b*TOTAL + m*output_size + y*W_out + x] =
            tex1Dfetch(images, b*input_size + y*28 + x) * filters[m*25]
            +tex1Dfetch(images, b*input_size + y*28 + x+1) * filters[m*25 + 1]
            +tex1Dfetch(images, b*input_size + y*28 + x+2) * filters[m*25 + 2]
            +tex1Dfetch(images, b*input_size + y*28 + x+3) * filters[m*25 + 3]
            +tex1Dfetch(images, b*input_size + y*28 + x+4) * filters[m*25 + 4]

            +tex1Dfetch(images, b*input_size + (y+1)*28 + x) * filters[m*25 + 5]
            +tex1Dfetch(images, b*input_size + (y+1)*28 + x+1) * filters[m*25 + 6]
            +tex1Dfetch(images, b*input_size + (y+1)*28 + x+2) * filters[m*25 + 7]
            +tex1Dfetch(images, b*input_size + (y+1)*28 + x+3) * filters[m*25 + 8]
            +tex1Dfetch(images, b*input_size + (y+1)*28 + x+4) * filters[m*25 + 9]

            +tex1Dfetch(images, b*input_size + (y+2)*28 + x) * filters[m*25 + 10]
            +tex1Dfetch(images, b*input_size + (y+2)*28 + x+1) * filters[m*25 + 11]
            +tex1Dfetch(images, b*input_size + (y+2)*28 + x+2) * filters[m*25 + 12]
            +tex1Dfetch(images, b*input_size + (y+2)*28 + x+3) * filters[m*25 + 13]
            +tex1Dfetch(images, b*input_size + (y+2)*28 + x+4) * filters[m*25 + 14]

            +tex1Dfetch(images, b*input_size + (y+3)*28 + x) * filters[m*25 + 15]
            +tex1Dfetch(images, b*input_size + (y+3)*28 + x+1) * filters[m*25 + 16]
            +tex1Dfetch(images, b*input_size + (y+3)*28 + x+2) * filters[m*25 + 17]
            +tex1Dfetch(images, b*input_size + (y+3)*28 + x+3) * filters[m*25 + 18]
            +tex1Dfetch(images, b*input_size + (y+3)*28 + x+4) * filters[m*25 + 19]

            +tex1Dfetch(images, b*input_size + (y+4)*28 + x) * filters[m*25 + 20]
            +tex1Dfetch(images, b*input_size + (y+4)*28 + x+1) * filters[m*25 + 21]
            +tex1Dfetch(images, b*input_size + (y+4)*28 + x+2) * filters[m*25 + 22]
            +tex1Dfetch(images, b*input_size + (y+4)*28 + x+3) * filters[m*25 + 23]
            +tex1Dfetch(images, b*input_size + (y+4)*28 + x+4) * filters[m*25 + 24];
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

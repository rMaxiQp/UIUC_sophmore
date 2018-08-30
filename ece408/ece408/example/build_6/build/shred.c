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

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x) {
    extern __shared__ float X_shared[];
    int counter, b, m, i, j, t_x = threadIdx.x, t_y = threadIdx.y, b_x = blockIdx.x;
    //float acc = 0;
    float * acc = &X_shared[input_size*(B/batch_size) + t_y*TILE_WIDTH + t_x];
    *acc = 0;

    for(b=b_x, counter=0; b<10000; b+=batch_size, counter++){
        for(i=t_y; i<28; i+=TILE_WIDTH){
            for(j=t_x; j<28; j+=TILE_WIDTH)
                X_shared[counter*input_size+ i*28 + j] = x[b*input_size + i*W + j];
        }
    }

    __syncthreads();

    for(b=b_x, counter=0; b<10000; b+=batch_size, counter++){
        for(m=0; m<50; m++){
            for(j=0; j<K; j++){
                for(i=0; i<K; i++)
                        *acc += X_shared[counter*input_size + (t_y+j)*28 + t_x+i] * filters[m*25 + j*K + i];
            }
            y[b*TOTAL + m*output_size + t_y*W_out + t_x] = *acc;
            *acc = 0;
        }
    }
}

template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {

    #ifdef EBM
        cudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
    #endif

    cudaMemcpyToSymbol(filters, k.dptr_, sizeof(float) * 1250);
    forward_kernel<gpu, DType><<<batch_size, dim3(TILE_WIDTH, TILE_WIDTH, 1), sizeof(float)*(input_size*(B/batch_size)) + sizeof(float) * 576, y.stream_->stream_>>>(y.dptr_,x.dptr_);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

}
}

#endif

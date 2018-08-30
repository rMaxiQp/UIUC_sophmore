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
    float acc;
    int counter, b, m, i, j, t_x = threadIdx.x, t_y = threadIdx.y, b_x = blockIdx.x;
    int offset2, offset3, offset4, offset5, offset6 = t_y*W_out;
    int offset = t_x*24+t_y;

    for(b=b_x, counter=0; b<10000; b+=batch_size, counter++){
        X_shared[counter*input_size + offset]=x[b*input_size + offset];
        if(offset<208)
            X_shared[counter*input_size + offset+576] = x[b*input_size + offset+576];
    }

    __syncthreads();

    for(b=b_x, counter=0; b<10000; b+=batch_size, counter++){
        offset = input_size * counter;
        offset5 = b*TOTAL;
        for(m=0; m<50; m++){
            acc = 0;
            offset3 = m*25;
            for(j=0; j<K; j++){
                offset2 = (t_y+j)*28;
                offset4 = j*K;
                for(i=0; i<K; i++)
                        acc += X_shared[offset + offset2 + t_x+i] * filters[offset3 + offset4 + i];
            }
            y[offset5 + m*output_size + offset6 + t_x] = acc;
        }
    }
}

template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {

    // #ifdef EBM
    //     cudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
    // #endif

    cudaMemcpyToSymbol(filters, k.dptr_, sizeof(float) * 1250);
    forward_kernel<gpu, DType><<<batch_size, dim3(TILE_WIDTH, TILE_WIDTH, 1), sizeof(float)*(input_size*(B/batch_size)), y.stream_->stream_>>>(y.dptr_,x.dptr_);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

}
}

#endif

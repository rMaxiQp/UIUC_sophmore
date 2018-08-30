
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// This function is called by new-inl.h
// Any code you write should be executed by this function
template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    // batch size
    const int B = x.shape_[0];
    // output chanel
    const int M = y.shape_[1];
    // input chanel
    const int C = x.shape_[1];
    // height
    const int H = x.shape_[2];
    // width
    const int W = x.shape_[3];
    // we use square kernel, so this is the kernel width
    const int K = k.shape_[3];

    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m){
            for (int h = 0; h < H-K+1; ++h){
                for (int w = 0; w < W-K+1; ++w){
                    y[b][m][h][w] = 0;
                    for (int c = 0; c < C; ++c){
                        for(int p = 0; p < K; ++p){
                            for(int q=0; q < K; ++q){
                                // ... a bunch of nested loops later...
                                y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                            }
                        }
                    }
                }
            }
        }
    }


}
}
}

#endif
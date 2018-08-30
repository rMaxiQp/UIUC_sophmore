
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
void forward(mshadow::Tensor<cpu, 4, DType> &y,
  const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    // k : [output feature map][input feature map][y][x]
    const int batch_size = x.shape_[0];
    const int input_channels = x.shape_[1]; // c
    const int input_y_size = x.shape_[2]; // h
    const int input_x_size = x.shape_[3]; // w
    const int output_channels = k.shape_[0];
    const int kernel_y = k.shape_[2];
    const int kernel_x = k.shape_[3];
    const int output_y_size = y.shape_[2];
    const int output_x_size = y.shape_[3];

    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
      for (int output_channel = 0; output_channel < output_channels; output_channel++) {
        // next we loop through all the output elements in a output channel
        for (int out_y = 0; out_y < output_y_size; out_y++) {
          for (int out_x = 0; out_x < output_x_size; out_x++) {
            y[batch_index][output_channel][out_y][out_x] = 0;
            // next we go through all the pixels related to y at this position
            for (int input_channel = 0; input_channel < input_channels; input_channel++) {
              for (int dy = 0; dy < kernel_y; dy++) {
                for (int dx = 0; dx < kernel_x; dx++) {
                  int yy = out_y + dy;
                  int xx = out_x + dx;
                  y[batch_index][output_channel][out_y][out_x] =
                    y[batch_index][output_channel][out_y][out_x] +
                    x[batch_index][input_channel][yy][xx] *
                    k[output_channel][input_channel][dy][dx];
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

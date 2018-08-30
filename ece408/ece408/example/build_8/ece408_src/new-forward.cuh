
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet {
  namespace op {

    //#define IN_IMAGE_SIZE 784
    //#define OUT_IMAGE_SIZE 576
    //#define IN_IMAGE_DIM 28
    //#define OUT_IMAGE_DIM 24
    #define NUM_FILTERS 50
    #define FILTER_SIZE 25
    __constant__ float k_const[NUM_FILTERS * FILTER_SIZE];

    __global__ void forward_kernel(float *y, const float *x) {

        /*
        Modify this function to implement the forward pass described in Chapter 16.
        We have added an additional dimension to the tensors to support an entire mini-batch
        The goal here is to be correct AND fast.
        We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
        */

        // An example use of these macros:
        // float a = y4d(0,0,0,0)
        // y4d(0,0,0,0) = a
        //for 1D thread block
        #define y3d(i2,i1,i0) y[(i2) * (28800) + (i1)*(576) + i0]
        #define x2d(i1,i0) x[(i1) * 784 + i0]

        int col = threadIdx.x % 24;
        int row = threadIdx.x / 24;

        //Load input image into shared memory
        __shared__ float image_sh[784];
        image_sh[threadIdx.x] = x2d(blockIdx.x,threadIdx.x);
        if ((threadIdx.x + blockDim.x) < (784))
          image_sh[threadIdx.x+blockDim.x] = x2d(blockIdx.x,threadIdx.x+blockDim.x);
        __syncthreads();
        /*
            Your code here!
        */
        int m;
        for (m = 0; m < 48; m += 12) { //Each thread block creates 50 output images
          //Create 12 output images at once
          float acc0 = 0.0; float acc1 = 0.0; float acc2 = 0.0; float acc3 = 0.0; float acc4 = 0.0;
          float acc5 = 0.0; float acc6 = 0.0; float acc7 = 0.0; float acc8 = 0.0; float acc9 = 0.0;
          float acc10 = 0.0; float acc11 = 0.0;
          int k_addr_idx0 = (m) * 25;
          int image_addr_idx = row * 28 + col;
          for (int p = 0; p < 5 ; ++p) {
            for (int q = 0; q < 5 ; ++q) {
              float in_elem = image_sh[image_addr_idx + q];
              acc0 += in_elem * k_const[k_addr_idx0];
              acc1 += in_elem * k_const[k_addr_idx0+25];
              acc2 += in_elem * k_const[k_addr_idx0+50];
              acc3 += in_elem * k_const[k_addr_idx0+75];
              acc4 += in_elem * k_const[k_addr_idx0+100];
              acc5 += in_elem * k_const[k_addr_idx0+125];
              acc6 += in_elem * k_const[k_addr_idx0+150];
              acc7 += in_elem * k_const[k_addr_idx0+175];
              acc8 += in_elem * k_const[k_addr_idx0+200];
              acc9 += in_elem * k_const[k_addr_idx0+225];
              acc10 += in_elem * k_const[k_addr_idx0+250];
              acc11 += in_elem * k_const[k_addr_idx0+275];
              k_addr_idx0++;
            }
            image_addr_idx += 28;
          }
          y3d(blockIdx.x,m, threadIdx.x) = acc0;
          y3d(blockIdx.x,m+1, threadIdx.x) = acc1;
          y3d(blockIdx.x,m+2, threadIdx.x) = acc2;
          y3d(blockIdx.x,m+3, threadIdx.x) = acc3;
          y3d(blockIdx.x,m+4, threadIdx.x) = acc4;
          y3d(blockIdx.x,m+5, threadIdx.x) = acc5;
          y3d(blockIdx.x,m+6, threadIdx.x) = acc6;
          y3d(blockIdx.x,m+7, threadIdx.x) = acc7;
          y3d(blockIdx.x,m+8, threadIdx.x) = acc8;
          y3d(blockIdx.x,m+9, threadIdx.x) = acc9;
          y3d(blockIdx.x,m+10, threadIdx.x) = acc10;
          y3d(blockIdx.x,m+11, threadIdx.x) = acc11;
        }
        //Can only iterate over 48 of 50 output images
        //Manually perform caculations for last two
        float acc48 = 0.0; float acc49 = 0.0;
        int k_addr_idx = m * 25;
        int image_addr_idx = row * 28 + col;
        for (int p = 0; p < 5 ; ++p) {
          for (int q = 0; q < 5 ; ++q) {
            float in_elem = image_sh[image_addr_idx + q];
            acc48 += in_elem * k_const[k_addr_idx];
            acc49 += in_elem * k_const[k_addr_idx+25];
            k_addr_idx++;
          }
          image_addr_idx += 28;
        }
        y3d(blockIdx.x,m, threadIdx.x) = acc48;
        y3d(blockIdx.x,m+1, threadIdx.x) = acc49;

        #undef y3d
        #undef x2d

    }

    /*
       This function is called by new-inl.h
       Any code you write should be executed by this function.
       For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
    */
    template<>
    void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {

        // You'll probably need to launch kernels against the right stream to keep MXNet happy
        cudaStream_t s = y.stream_->stream_;

        // Extract the tensor dimensions into B
        // const int B = x.shape_[0];

        float *k_host;
        unsigned int k_size = NUM_FILTERS * FILTER_SIZE * sizeof(float);
        k_host = (float *) malloc(k_size);
        cudaMemcpy (k_host, w.dptr_, k_size, cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol (k_const, k_host, k_size);

        // Set the kernel dimensions
        //gridDim.y = images in a batch (10,000)
        //gridDim.x = images in an input (1)
        dim3 gridDim (x.shape_[0], 1, 1);
        dim3 blockDim (576, 1, 1); //blockDim will be size of output image

        // Call the kernel
        forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_);

        // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        free (k_host);
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

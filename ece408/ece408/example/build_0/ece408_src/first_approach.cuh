#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH_UNROLL 576 // Block dimension used for unrolling the matrices
#define TILE_WIDTH 25 // Tile width for doing the tiled matrix multiplication


#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//__global__ void forward_kernel(DType *y, const DType *x, const DType *k, float* filter_unroll,const int B, const int M, const int C, const int H, const int W, const int K) {
template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, DType *x, const DType *filter,const int M, const int C, const int H, const int W, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    //#define k4d(i3,i2,i1,i0) filter[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    __shared__ float ds_filter[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_x[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by*blockDim.y + ty;
    int Col = bx*blockDim.x + tx;

    int filter_width = C*K*K;
    //int filter_height = M;
    int x_width = H_out*W_out;
    int x_height = C*K*K;

    int b = blockIdx.z;
    int offset = b*x_width*x_height;

    float result = 0.0;

    for (int i = 0; i < ((filter_width-1)/TILE_WIDTH)+1;i++){
        //if ((Row < filter_height) && (i*TILE_WIDTH+tx < filter_width)){
        //    ds_filter[ty][tx] = filter[Row*filter_width + i*TILE_WIDTH+tx];
        //}
        //else{
        //    ds_filter[ty][tx] = 0.0;
        //}
        ds_filter[ty][tx] = filter[Row*filter_width + i*TILE_WIDTH+tx];
        if ((Col < x_width)  && (i*TILE_WIDTH+ty < x_height)){
            ds_x[ty][tx] = x[(i*TILE_WIDTH+ty)*x_width + Col + offset];
        }
        else{
            ds_x[ty][tx] = 0.0;
        }
        __syncthreads();


        int t = 0;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 1;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 2;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 3;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 4;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 5;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 6;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 7;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 8;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 9;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 10;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 11;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 12;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 13;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 14;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 15;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 16;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 17;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 18;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 19;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 20;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 21;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 22;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 23;
        result += ds_filter[ty][t]*ds_x[t][tx];
        t = 24;
        result += ds_filter[ty][t]*ds_x[t][tx];

        __syncthreads();
    }

    if (Row < M && Col < W_out*H_out){
        int output_row = Col/W_out; // row index of an  output element in the output matrix
        int output_col = Col%W_out; // col index of an output element in the output matrix
        y4d(b,Row,output_row,output_col) = result;
    }


/*
    int offset = 20*b*x_width*x_height;
    for (int i = 0; i < ((filter_width-1)/TILE_WIDTH)+1;i++){
        if ((Row < filter_height) && (i*TILE_WIDTH+tx < filter_width)){
            ds_filter[ty][tx] = filter[Row*filter_width + i*TILE_WIDTH+tx];
        }
        else{
            ds_filter[ty][tx] = 0.0;
        }
        if ((Col < x_width)  && (i*TILE_WIDTH+ty < x_height)){
            ds_x[ty][tx] = x[(i*TILE_WIDTH+ty)*x_width + Col + offset];
        }
        else{
            ds_x[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int t =0; t < TILE_WIDTH; t++){
            result += ds_filter[ty][t]*ds_x[t][tx];
        }
        __syncthreads();
    }

    if (Row < M && Col < W_out*H_out){
        int output_row = Col/W_out; // row index of an  output element in the output matrix
        int output_col = Col%W_out; // col index of an output element in the output matrix
        y4d(20*b,Row,output_row,output_col) = result;
    }
*/

    #undef y4d
    #undef x4d
    //#undef k4d
}

template<typename gpu, typename DType>
__global__ void x_unroll(int C,int H, int W,int K, const DType *x, DType* X_unroll){
    int c, s, col_out, row_out, col_unroll, row_unroll,row_base, p, q,idx;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out*W_out;
    int H_unroll = C*K*K;

    int b = blockIdx.z;
    int offset = b*W_unroll*H_unroll;

    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    if (t < C*W_unroll){
        c = t/W_unroll; // index of input channel
        s = t%W_unroll; // linearized index of each output element
        row_out = s/W_out; // row index of each output element 
        col_out = s%W_out; // col index of each output element
        col_unroll = row_out*W_out + col_out;   // Col index in unrolled x matrix (should just be equal to s???)
        row_base = c*K*K;
        for (p=0;p<K;p++){
            for (q=0;q<K;q++){
                    row_unroll = row_base + p*K+q; // row index in unrolled x matrix
                    idx = row_unroll*(H_out*W_out) + col_unroll; // linearized index in the unrolled x matrix
                    //X_unroll[idx] = X[b][c][row_out+p][col_out+q];
                    X_unroll[idx+offset] = x4d(b,c,row_out+p,col_out+q);
            }
        }
    }
    #undef x4d

}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    //You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; // batch size
    const int M = y.shape_[1]; // Number of output channels
    const int C = x.shape_[1]; // Number of input channels
    const int H = x.shape_[2]; // height of each input channel
    const int W = x.shape_[3]; // width of each input channel
    const int K = w.shape_[3]; // size of the square kernel

    int W_out = W-K+1;
    int H_out = H-K+1;


    // Buffer for unrolled X matrix of each image in minibatch
    int H_unroll = C*K*K;
    int W_unroll = H_out * W_out;
    DType* X_unroll;
    cudaMalloc((void **) &X_unroll,B*W_unroll*H_unroll*sizeof(DType));

    dim3 x_unrollGrid((C*H_out*W_out-1)/TILE_WIDTH_UNROLL+1,1,B);
    dim3 x_unrollBlock(TILE_WIDTH_UNROLL,1,1);
    x_unroll<gpu,DType><<<x_unrollGrid,x_unrollBlock,0,s>>>(C,H,W,K,x.dptr_,X_unroll);

    //int Tile_width = 16;
    dim3 Grid((H_out*W_out-1)/TILE_WIDTH+1,(M-1)/TILE_WIDTH +1,B/1);
    dim3 Block(TILE_WIDTH,TILE_WIDTH,1);

    //forward_kernel<gpu,DType><<<Grid,Block,0,s>>>(y.dptr_,X_unroll,w.dptr_,M,C,H,W,K);
    forward_kernel<gpu,DType><<<Grid,Block,0,s>>>(y.dptr_,X_unroll,w.dptr_,M,C,H,W,K);


    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    //cudaFree(filter_Unroll);
    cudaFree(X_unroll);

}



}
}

#endif

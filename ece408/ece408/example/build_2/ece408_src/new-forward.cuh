
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>


namespace mxnet
{
namespace op
{


__global__ void matrixShift(float*x, float*y, int C,int H,int W,int K,int B){
    
    int t = threadIdx.x;
    int start = blockIdx.x*24*24*50;
    int index = blockIdx.y;
    float temp1=0.0f;
    float temp2=0.0f;

 
    
        temp1 = x[blockIdx.x*24*24+t+index*24*24*B];
        temp2 = x[blockIdx.x*24*24+t+288+index*24*24*B];
    

        y[start+index*24*24+t] = temp1;
        y[start+index*24*24+t+288] = temp2;
         
         

       
    }






__constant__ float ABC[50*25];
__global__ void matrixMultiplyShared2(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

    #define TILE_WIDTH 32

 
  extern __shared__ float shmem[];
  float *Mds = &shmem[0];
float *Nds = &shmem[TILE_WIDTH*TILE_WIDTH];
   int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
      float Pvalue0 =0.0f;
      float Pvalue1 = 0.0f;
      float Pvalue2 = 0.0f;
      float Pvalue3 = 0.0f;
   
   int Row = by * blockDim.y + ty;
    int Col =bx*blockDim.x*4+tx;
  int m=0;


 
       if( (Row<numARows)&&(tx<numAColumns) )
         Mds[ty*TILE_WIDTH+tx] = ABC[Row*numAColumns+tx];
     else
         Mds[ty*TILE_WIDTH+tx] =0.0;
     if( (ty<numBRows) && (Col<numBColumns) )
         Nds[ty*TILE_WIDTH+tx] = B[ty*numBColumns+Col];
     else
         Nds[ty*TILE_WIDTH+tx]= 0.0;

          if( (Row<numARows)&&( (tx+TILE_WIDTH/4)<numAColumns) )
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH/4] = ABC[Row*numAColumns+tx+TILE_WIDTH/4];
     else
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH/4] =0.0;
     if( (ty<numBRows) && ( (Col+TILE_WIDTH/4)<numBColumns) )
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH/4] = B[ty*numBColumns+Col+TILE_WIDTH/4];
     else
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH/4]= 0.0;


    if( (Row<numARows)&&( (tx+TILE_WIDTH/2)<numAColumns) )
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH/2] = ABC[Row*numAColumns+tx+TILE_WIDTH/2];
     else
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH/2] =0.0;
     if( (ty<numBRows) && ( (Col+TILE_WIDTH/2)<numBColumns) )
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH/2] = B[ty*numBColumns+Col+TILE_WIDTH/2];
     else
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH/2]= 0.0;


         if( (Row<numARows)&&( (tx+TILE_WIDTH*3/4)<numAColumns) )
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH*3/4] = ABC[Row*numAColumns+tx+TILE_WIDTH*3/4];
     else
         Mds[ty*TILE_WIDTH+tx+TILE_WIDTH*3/4] =0.0;
     if( (ty<numBRows) && ( (Col+TILE_WIDTH*3/4)<numBColumns) )
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH*3/4] = B[ty*numBColumns+Col+TILE_WIDTH*3/4];
     else
         Nds[ty*TILE_WIDTH+tx+TILE_WIDTH*3/4]= 0.0;



     
       


     __syncthreads();

     if(Row<numCRows && Col < numCColumns){
        #pragma unroll 
     for(m=0;m<TILE_WIDTH;m++){
       Pvalue0 += Mds[(ty+0)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue1 += Mds[(ty)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx+TILE_WIDTH/4];
       Pvalue2 += Mds[(ty)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx+TILE_WIDTH/2];
       Pvalue3 += Mds[(ty)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx+TILE_WIDTH*3/4];

     }
     
      
   }
    __syncthreads();
   
     
      if(Row<numCRows && Col < numCColumns) C[Row*numCColumns+Col] = Pvalue0;
      if( (Row)<numCRows && (Col+TILE_WIDTH/4) < numCColumns) C[(Row)*numCColumns+Col+TILE_WIDTH/4] = Pvalue1;
       if( (Row)<numCRows && (Col+TILE_WIDTH/2) < numCColumns) C[(Row)*numCColumns+Col+TILE_WIDTH/2] = Pvalue2;
        if( (Row)<numCRows && (Col+TILE_WIDTH*3/4) < numCColumns) C[(Row)*numCColumns+Col+TILE_WIDTH*3/4] = Pvalue3;
   
 #undef TILE_WIDTH
}



__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {



  int TILE_WIDTH = 32;
  extern __shared__ float shmem[];
  float *Mds = &shmem[0];
float *Nds = &shmem[TILE_WIDTH*TILE_WIDTH];
   int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
      float Pvalue0 =0.0;
      float Pvalue1 =0.0;
      float Pvalue2 = 0.0;
      float Pvalue3 = 0.0;
      float Pvalue4 = 0.0;
      float Pvalue5 = 0.0;
      float Pvalue6 = 0.0;
      float Pvalue7 = 0.0;
      float Pvalue8 = 0.0;
      float Pvalue9 = 0.0;
      float Pvalue10 = 0.0;
      float Pvalue11 = 0.0;
      float Pvalue12 = 0.0;
      float Pvalue13 = 0.0;
      float Pvalue14 = 0.0;
      float Pvalue15 = 0.0;
      float Pvalue16 = 0.0;
      float Pvalue17 = 0.0;
      float Pvalue18 = 0.0;
      float Pvalue19 = 0.0;
      float Pvalue20 = 0.0;
      float Pvalue21 = 0.0;
      float Pvalue22 = 0.0;
      float Pvalue23 = 0.0;
      float Pvalue24 = 0.0;
      float Pvalue25 = 0.0;
      float Pvalue26 = 0.0;
      float Pvalue27 = 0.0;
      float Pvalue28 = 0.0;
      float Pvalue29 = 0.0;
      float Pvalue30 = 0.0;
      float Pvalue31 = 0.0;
   int Row = by * blockDim.y*32 + ty;
    int Col =bx*blockDim.x+tx;
  int k=0;
  int m=0;
   for(k=0;k<ceil(numAColumns/(float)TILE_WIDTH);k++)
   {
        #pragma unroll
        for(int loop=0;loop<32;loop++){
            Mds[(ty+loop)*TILE_WIDTH+tx]=0.0f;
            Nds[(ty+loop)*TILE_WIDTH+tx]=0.0f;
                    

     if(( (Row+loop)<numARows)&&(k*TILE_WIDTH+tx)<numAColumns)
         Mds[ (ty+loop)*TILE_WIDTH+tx] = ABC[(Row+loop)*numAColumns + k*TILE_WIDTH +tx];

     if((k*TILE_WIDTH+ty+loop)<numBRows && Col<numBColumns)
         Nds[(ty+loop)*TILE_WIDTH+tx] = B[(k*TILE_WIDTH+ty+loop)*numBColumns+Col];

        }



     __syncthreads();

     if(Row<numCRows && Col < numCColumns){
        #pragma unroll 
     for(m=0;m<TILE_WIDTH;m++){
       Pvalue0 += Mds[(ty+0)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue1 +=Mds[ (ty+1) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue2 += Mds[ (ty+2) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue3 += Mds[ (ty+3) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue4 += Mds[(ty+4)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue5 +=Mds[ (ty+5) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue6 += Mds[ (ty+6) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue7 += Mds[ (ty+7) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
        Pvalue8 += Mds[(ty+8)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue9 +=Mds[ (ty+9) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue10 += Mds[ (ty+10) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue11 += Mds[ (ty+11) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue12 += Mds[(ty+12)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue13 +=Mds[ (ty+13) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue14 += Mds[ (ty+14) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue15 += Mds[ (ty+15) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
        Pvalue16 += Mds[(ty+16)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue17 +=Mds[ (ty+17) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue18 += Mds[ (ty+18) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue19 += Mds[ (ty+19) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue20 += Mds[(ty+20)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue21 +=Mds[ (ty+21) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue22 += Mds[ (ty+22) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue23 += Mds[ (ty+23) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
        Pvalue24 += Mds[(ty+24)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue25 +=Mds[ (ty+25) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue26 += Mds[ (ty+26) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue27 += Mds[ (ty+27) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue28 += Mds[(ty+28)*TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue29 +=Mds[ (ty+29) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue30 += Mds[ (ty+30) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
       Pvalue31 += Mds[ (ty+31) *TILE_WIDTH+m] * Nds[m*TILE_WIDTH+tx];
        
     }
     
      
   }
    __syncthreads();
   }
     
      if(Row<numCRows && Col < numCColumns) C[Row*numCColumns+Col] = Pvalue0;
      if( (Row+1)<numCRows)  C[(Row+1)*numCColumns+Col] = Pvalue1;
      if( (Row+2)<numCRows)  C[(Row+2)*numCColumns+Col] = Pvalue2;
      if( (Row+3)<numCRows)  C[(Row+3)*numCColumns+Col] = Pvalue3;
      if( (Row+4)<numCRows)  C[(Row+4)*numCColumns+Col] = Pvalue4;
      if( (Row+5)<numCRows)  C[(Row+5)*numCColumns+Col] = Pvalue5;
      if( (Row+6)<numCRows)  C[(Row+6)*numCColumns+Col] = Pvalue6;
      if( (Row+7)<numCRows)  C[(Row+7)*numCColumns+Col] = Pvalue7;
      if( (Row+8)<numCRows)  C[(Row+8)*numCColumns+Col] = Pvalue8;
      if( (Row+9)<numCRows)  C[(Row+9)*numCColumns+Col] = Pvalue9;
      if( (Row+10)<numCRows)  C[(Row+10)*numCColumns+Col] = Pvalue10;
      if( (Row+11)<numCRows)  C[(Row+11)*numCColumns+Col] = Pvalue11;
      if( (Row+12)<numCRows)  C[(Row+12)*numCColumns+Col] = Pvalue12;
      if( (Row+13)<numCRows)  C[(Row+13)*numCColumns+Col] = Pvalue13;
      if( (Row+14)<numCRows)  C[(Row+14)*numCColumns+Col] = Pvalue14;
      if( (Row+15)<numCRows)  C[(Row+15)*numCColumns+Col] = Pvalue15;
      if( (Row+16)<numCRows)  C[(Row+16)*numCColumns+Col] = Pvalue16;
      if( (Row+17)<numCRows)  C[(Row+17)*numCColumns+Col] = Pvalue17;
      if( (Row+18)<numCRows)  C[(Row+18)*numCColumns+Col] = Pvalue18;
      if( (Row+19)<numCRows)  C[(Row+19)*numCColumns+Col] = Pvalue19;
      if( (Row+20)<numCRows)  C[(Row+20)*numCColumns+Col] = Pvalue20;
      if( (Row+21)<numCRows)  C[(Row+21)*numCColumns+Col] = Pvalue21;
      if( (Row+22)<numCRows)  C[(Row+22)*numCColumns+Col] = Pvalue22;
      if( (Row+23)<numCRows)  C[(Row+23)*numCColumns+Col] = Pvalue23;
      if( (Row+24)<numCRows)  C[(Row+24)*numCColumns+Col] = Pvalue24;
      if( (Row+25)<numCRows)  C[(Row+25)*numCColumns+Col] = Pvalue25;
      if( (Row+26)<numCRows)  C[(Row+26)*numCColumns+Col] = Pvalue26;
      if( (Row+27)<numCRows)  C[(Row+27)*numCColumns+Col] = Pvalue27;
      if( (Row+28)<numCRows)  C[(Row+28)*numCColumns+Col] = Pvalue28;
      if( (Row+29)<numCRows)  C[(Row+29)*numCColumns+Col] = Pvalue29;
      if( (Row+30)<numCRows)  C[(Row+30)*numCColumns+Col] = Pvalue30;
      if( (Row+31)<numCRows)  C[(Row+31)*numCColumns+Col] = Pvalue31;

          
      
      
  
}



__global__ void matrixMultiply(float *w, float *x, float *y, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {


  //@@ Insert code to implement matrix multiplication here
  int rows = threadIdx.y + blockDim.y*blockIdx.y;
  int cols = threadIdx.x + blockDim.x*blockIdx.x;
  int k=0;
  if(rows < numCRows && cols < numCColumns)
  {
    float value = 0.0;
    for(k=0;k<numBRows;k++){
      value = value + w[rows*numAColumns+k]*x[k*numBColumns+cols];
    }
    y[rows*numCColumns+cols]= value;
   
    
  }
  
}

__global__ void unroll_kernel(int C,int H,int W,int K,int B,float* x,float* x_unroll)
{
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

int h_out,w_out,p,q;
const int H_out = H-K+1;
const int W_out = W-K+1;
const int W_unroll = H_out*W_out;

if(threadIdx.x<W_unroll){
    
    h_out = threadIdx.x/W_out;
    w_out = threadIdx.x%W_out;
    for(p=0;p<K;p++)
        for(q=0;q<K;q++){
            x_unroll[(p*K+q)*W_unroll*B+threadIdx.x+blockIdx.x*H_out*W_out] = x4d(blockIdx.x,0,h_out+p,w_out+q);
            
        }



}

 #undef x4d
}




__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

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

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */


      
/*
int W_grid = (W_out-1)/12 +1; 
int n, m, h, w, c, p, q;
n = blockIdx.x;
m = blockIdx.y;
h = (blockIdx.z/W_grid)*12 +threadIdx.y; 
w = (blockIdx.z%W_grid)*12 + threadIdx.x; 
float acc = 0.;
for(c=0; c<C;c++){ //sumoverallinputchannels 
    for(p=0;p<K;p++){ //loopoverKxK filter
        for (q = 0; q < K; q++){
            acc = acc + x4d(n,c,h+p,w+q)*k4d(m,c,p,q);
            }
    }

    }
    y4d(n,m,h,w)= acc; 
*/

int W_grid = (W_out-1)/24 +1;
int n,m,h0,w0,h_base,w_base,h,w;
int tile_width = 24;
int X_tile_width = tile_width+K-1;
extern __shared__ float shmem[];
float *X_shared = &shmem[0];
float *W_shared = &shmem[X_tile_width*X_tile_width];
n = blockIdx.x;
m = blockIdx.y;
h0 = threadIdx.x;
w0 = threadIdx.y;
h_base = (blockIdx.z/W_grid) *tile_width;
w_base = (blockIdx.z%W_grid) *tile_width;
h = h_base+h0;
w = w_base+w0;

float acc = 0;
int c,p,q;
for(c=0;c<C;c++){
    if( (h0<K) && (w0<K) )
        W_shared[h0*K+w0] = k4d(m,c,h0,w0);
    __syncthreads();

    for(int i=h;i<h_base+X_tile_width;i+=tile_width){
        for(int j=w;j<w_base+X_tile_width;j+=tile_width)
            X_shared[ (i-h_base)*X_tile_width+j-w_base] = x4d(n,c,i,j);

    }
    __syncthreads();
    for(p=0;p<K;p++){
        for(q=0;q<K;q++)
        acc = acc + X_shared[ (h+p)*X_tile_width+w+q] * W_shared[p*K+q];

    }
    __syncthreads();
    y4d(n,m,h,w) = acc;



}






    #undef y4d
    #undef x4d
    #undef k4d
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
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
     cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    

   const int B = x.shape_[0];
    int M = y.shape_[1];
    int C = x.shape_[1];
    int H = x.shape_[2];
    int W = x.shape_[3];
    int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    //int W_grid = (W_out-1)/24 +1;
    //int H_grid = (H_out-1)/24 +1;

 
    //int z = H_grid * W_grid;
    
     //Set the kernel dimensions
    //dim3 gridDim(B,M,z);
    //dim3 blockDim(24,24,1);
    //size_t shmem_size = sizeof(float) * ( (24+ K-1)*(24+ K-1) + K*K ); 

     //Call the kernel
     //forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

        // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        //return;
     

      
     
     /*
     int W_unroll = H_out*W_out;
     int H_unroll = C*K*K;
     float* X_unrolled;
     float* y_unshift;
     cudaMalloc((void**)&X_unrolled,W_unroll *H_unroll*B*sizeof(float) );
     cudaMalloc((void**)&y_unshift, B*W_unroll*M*sizeof(float));
    
     dim3 dimGrid1( ceil( (W_unroll*B)/16.0), ceil(M/16.0),1);
     dim3 dimBlock1(16,16,1);
    float* y_index;
     dim3 dimGrid_unroll(B,1,1);
     dim3 dimBlock_unroll(H_out*W_out,1,1);

    
     
    unroll_kernel<<<dimGrid_unroll,dimBlock_unroll,0,s>>>(C,H, W,K,B,x.dptr_,X_unrolled);
     matrixMultiply<<<dimGrid1,dimBlock1,0,s>>>(w.dptr_,X_unrolled,y_unshift,M,H_unroll,H_unroll,W_unroll*B,M,W_unroll*B);
     matrixShift<<<B,24*24/4,0,s>>>(y_unshift, y.dptr_, C, H, W, K,B);
     
     */


     
     int W_unroll = H_out*W_out;
     int H_unroll = C*K*K;
     float* X_unrolled;
     float* y_unshift;
     cudaMalloc((void**)&X_unrolled,B*W_unroll *H_unroll*sizeof(float) );
     cudaMalloc((void**)&y_unshift, B*W_unroll*M*sizeof(float));
    
     
     cudaMemcpyToSymbol(ABC, w.dptr_, sizeof(float)*5*5*50 );

     int tile_width=32;
     dim3 dimGrid2( ceil( (W_unroll*B)/32.0), ceil(M/32.0),1);
     dim3 dimBlock2(tile_width/4,tile_width,1);
     size_t shmem_size2 = sizeof(float) * 32*32*2 ; 
     


     dim3 dimGrid_unroll(B,1,1);
     dim3 dimBlock_unroll(576,1,1);

     dim3 dimGrid_shift(B,50,1);
     dim3 dimBlock_shift(24*12,1,1);
  
    unroll_kernel<<<dimGrid_unroll,dimBlock_unroll,0,s>>>(C,H, W,K,B,x.dptr_,X_unrolled);
    matrixMultiplyShared2<<<dimGrid2,dimBlock2,shmem_size2,s>>>(w.dptr_,X_unrolled,y_unshift,M,H_unroll,H_unroll,W_unroll*B,M,W_unroll*B);


     
    matrixShift<<<dimGrid_shift,dimBlock_shift,0,s>>>(y_unshift, y.dptr_, C, H, W, K,B);
    
    

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

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
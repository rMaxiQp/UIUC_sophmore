// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *aux) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  __shared__ float destination[BLOCK_SIZE];
  __shared__ float source[BLOCK_SIZE];
  
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int tid = threadIdx.x;
  
  //move to shared mem
  if(idx < len)
    source[tid] = input[idx];
  else
    source[tid] = 0.0;
  
  destination[tid] = source[tid];
  
  __syncthreads();

  //scan
  for(int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    
    if(tid >= stride)
      destination[tid] = source[tid] + source[tid - stride];
    __syncthreads();
    source[tid] = destination[tid];
    __syncthreads();
  }

  //move to output
  if(idx < len)
    output[idx] = source[tid];
  
  if(tid == BLOCK_SIZE - 1)
    aux[blockIdx.x] = source[tid];
}

__global__ void sum_up(float *output, float *aux, int len) {
  int bx = blockIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  float value = 0.0;
  if(idx < len)
    value = output[idx];
  
  for(int i = 0; i < bx; i ++)
    value += aux[i];
  
  if(idx < len)
    output[idx] = value;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *aux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux, ((numElements-1) / BLOCK_SIZE + 1) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(aux, 0, ((numElements-1) / BLOCK_SIZE + 1) * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(((numElements-1) / BLOCK_SIZE + 1), 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, aux);
  cudaDeviceSynchronize();
  sum_up<<<dimGrid, dimBlock>>>(deviceOutput, aux, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}


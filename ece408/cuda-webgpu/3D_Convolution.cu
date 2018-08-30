#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define Tile_Width 8
#define Mask_Width  3
#define Mask_Radius  1
#define Mem_Width (Tile_Width + Mask_Width - 1)

//@@ Define constant memory for device kernel here
__constant__ float Mask[Mask_Width][Mask_Width][Mask_Width];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N[Mem_Width][Mem_Width][Mem_Width];
  
  int x = threadIdx.x + (blockIdx.x * Tile_Width);
  int y = threadIdx.y + (blockIdx.y * Tile_Width);
  int z = threadIdx.z + (blockIdx.z * Tile_Width);
  
  int N_z = z - Mask_Radius;
  int N_y = y - Mask_Radius;
  int N_x = x - Mask_Radius;
  
  if(N_z >= 0 && N_z < z_size && N_y >= 0 && N_y < y_size && N_x >= 0 && N_x < x_size)
    N[threadIdx.z][threadIdx.y][threadIdx.x] = input[N_z * (y_size * x_size) + N_y * (x_size) + N_x];
  else
    N[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  
  __syncthreads();
  float res = 0.0f;
  if(threadIdx.z < Tile_Width && threadIdx.y < Tile_Width && threadIdx.x < Tile_Width) {
   /* for(int z_mask = -Mask_Radius; z_mask <= Mask_Radius; z_mask++){
      for(int y_mask = -Mask_Radius; y_mask <= Mask_Radius; y_mask++){
        for(int x_mask = -Mask_Radius; x_mask <= Mask_Radius; x_mask++){
          res += Mask[z_mask + Mask_Radius][y_mask + Mask_Radius][x_mask + Mask_Radius] 
                  * N[z_mask + Mask_Radius + threadIdx.z][y_mask + Mask_Radius + threadIdx.y][x_mask + Mask_Radius + threadIdx.x];//input[z_in * (y_size * x_size) + y_in * (x_size) + x_in];
        }
      }
    }*/
     for(int z_mask = 0; z_mask < Mask_Width; z_mask++){
      for(int y_mask = 0; y_mask < Mask_Width; y_mask++){
        for(int x_mask = 0; x_mask < Mask_Width; x_mask++){
          res += Mask[z_mask][y_mask][x_mask ] 
                  * N[z_mask + threadIdx.z][y_mask  + threadIdx.y][x_mask + threadIdx.x];//input[z_in * (y_size * x_size) + y_in * (x_size) + x_in];
        }
      }
    }
      __syncthreads();
  if(z < z_size && y <y_size && x < x_size)
    output[z * (y_size * x_size) + y * (x_size) + x] = res;
  }


}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc(&deviceInput, z_size * y_size * x_size * sizeof(float));
  cudaMalloc(&deviceOutput, z_size * y_size * x_size * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mask,hostKernel, Mask_Width * Mask_Width * Mask_Width * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  dim3 dimBlock(10,10, 10);
  dim3 dimGrid(ceil(x_size/8.0), ceil(y_size/8.0), ceil(z_size/8.0));
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}


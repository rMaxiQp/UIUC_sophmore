// Histogram Equalization

#include <wb.h>
#include <stdio.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512
//@@ insert code here

/*
  for ii from 0 to (width * height * channels) do
    ucharImage[ii] = (unsigned char) (255 * inputImage[ii])
  end
 */
__global__ void to_int(float *in, unsigned char *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(idx < total)
    out[idx] = (unsigned char) (255 * in[idx]);
}

/*
  for ii from 0 to height do
    for jj from 0 to width do
        idx = ii * width + jj
        # here channels is 3
        r = ucharImage[3*idx]
        g = ucharImage[3*idx + 1]
        b = ucharImage[3*idx + 2]
        grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)
    end
  end
 */

__global__ void to_gray(unsigned char *in, unsigned char *out, int len, int imageChannels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < len) {
    unsigned char r = in[imageChannels * idx];
    unsigned char g = in[imageChannels * idx + 1];
    unsigned char b = in[imageChannels * idx + 2];
    out[idx] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
  
}

/*
  histogram = [0, ...., 0] # here len(histogram) = 256
   for ii from 0 to width * height do
     histogram[grayImage[ii]]++
  end
 */
__global__ void histogram(unsigned char *in, unsigned int *hist, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < len)
    atomicAdd(&hist[in[idx]], 1);
}

/*
  def p(x):
    return x / (width * height)
  end
  
  cdf[0] = p(histogram[0])
    for ii from 1 to 256 do
      cdf[ii] = cdf[ii - 1] + p(histogram[ii])
  end
 */
__global__ void to_cdf(float *cdf, float *mincdf, unsigned int *hist, int size) {
  __shared__ float reducer[HISTOGRAM_LENGTH];
  int tid = threadIdx.x;
  int idx = HISTOGRAM_LENGTH * blockIdx.x + threadIdx.x;
  int half = HISTOGRAM_LENGTH / 2;
  
  if(idx < HISTOGRAM_LENGTH)
    reducer[tid] = hist[idx] / (1.0 * size); //p(hist[idx])
  else
    reducer[tid] = 0.0;

  if(idx < half)
    reducer[tid + half] = hist[idx + half] / (1.0 * size);
  else
    reducer[tid + half] = 0.0;
    
  for(int stride = 1; stride <= half; stride *= 2) {
    __syncthreads();
    int t = 2 * (tid + 1) * stride - 1;
    if(t < HISTOGRAM_LENGTH)
      reducer[t] += reducer[t - stride];
  }
  
  for(int stride = half/2; stride >= 1; stride /= 2) {
    __syncthreads();
    int t = 2 * (tid + 1) * stride - 1;
    if(t + stride < HISTOGRAM_LENGTH)
      reducer[t + stride] += reducer[t];
  }
  
  __syncthreads();
  
  if(idx < HISTOGRAM_LENGTH)
    cdf[idx] = reducer[tid];
  if(idx < half)
    cdf[idx + half] = reducer[tid + half];

  if(idx == 0)
    mincdf[0] = cdf[0];
}

/*
  def correct_color(val) 
    return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
  end

  def clamp(x, start, end)
      return min(max(x, start), end)
  end

  for ii from 0 to (width * height * channels) do
    ucharImage[ii] = correct_color(ucharImage[ii])
  end
  */
__global__ void equalization(unsigned char *image, float *cdf, float *mincdf, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < total) {
    float f = (255.0 * cdf[image[idx]] - *mincdf) / (1 - *mincdf);
    
    if (f < 0.0)
      f = 0.0;
    else if(f > 255.0)
      f = 255.0;
    
    image[idx] = (unsigned char) f;
  }
}

/*
  for ii from 0 to (width * height * channels) do
    outputImage[ii] = (float) (ucharImage[ii]/255.0)
  end
 */

__global__ void to_float(unsigned char *in, float *out, int total) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(idx < total)
    out[idx] = (float) (in[idx] / 255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  
  float *deviceInput;
  float *deviceOutput;
  unsigned char *intImage;
  unsigned char *gray;
  unsigned int *hist;
  float *cdf;
  float *mincdf;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int len = imageWidth * imageHeight;
  int total = len * imageChannels;
  
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **)&deviceInput, total * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, total * sizeof(float)));
  wbCheck(cudaMalloc((void **)&intImage, total * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&gray, imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&hist, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&mincdf, sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");


  //@@ Copy input and kernel to GPU here
  wbCheck(cudaMemcpy(deviceInput, hostInputImageData, total * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(hist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  
  wbLog(TRACE, "The value of imageWidth = ", imageWidth);
  wbLog(TRACE, "The value of imageHeight = ", imageHeight);
  wbLog(TRACE, "The value of imageChannels = ", imageChannels);
  
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((total - 1) / BLOCK_SIZE + 1, 1, 1);
  
  dim3 dimBlockInner(BLOCK_SIZE, 1, 1);
  dim3 dimGridInner((imageHeight * imageWidth - 1) / BLOCK_SIZE + 1, 1, 1);
  
  dim3 dimBlockCDF(HISTOGRAM_LENGTH / 2, 1, 1);
  dim3 dimGridCDF(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU kernel here
  
  //float *in, unsigned char *out, int total
  to_int<<<dimGrid, dimBlock>>>(deviceInput, intImage, total);
  cudaDeviceSynchronize();
  
  //unsigned char *in, unsigned char *out, int len, int imageChannels
  to_gray<<<dimGridInner, dimBlockInner>>>(intImage, gray, len, imageChannels);
  cudaDeviceSynchronize();

  /*
  unsigned char *debug_gray = (unsigned char *) malloc(imageWidth * imageHeight * sizeof(unsigned char));
  wbCheck(cudaMemcpy(debug_gray, gray, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 256; i++)
    fprintf(stderr, "debug_gray %d: %c\n", i, debug_gray[i]);
  */

  //unsigned char *in, unsigned int *hist, int len
  histogram<<<dimGridInner, dimBlockInner>>>(gray, hist, len);
  cudaDeviceSynchronize();
  
  /*
  unsigned int* debug_hist = (unsigned int*)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  wbCheck(cudaMemcpy(debug_hist, hist, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 256; i++)
    fprintf(stderr, "debug_hist %d: %d\n", i, debug_hist[i]);
  */
   
  //float *cdf, float *mincdf, unsigned int *hist, int size
  to_cdf<<<dimGridCDF, dimBlockCDF>>>(cdf, mincdf, hist, len);
  cudaDeviceSynchronize();

  /*
  float* debug_cdf = (float*)malloc(HISTOGRAM_LENGTH * sizeof(float));
  wbCheck(cudaMemcpy(debug_cdf, cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 256; i++)
    fprintf(stderr, "debug_cdf %d: %f\n", i, debug_cdf[i]);
  */
  
  //unsigned char *image, float *cdf, float *mincdf, int total
  equalization<<<dimGrid, dimBlock>>>(intImage, cdf, mincdf, total);
  cudaDeviceSynchronize();
  
  /*
  unsigned char* debug_image = (unsigned char*)malloc(total * sizeof(unsigned char));
  wbCheck(cudaMemcpy(debug_image, intImage, total * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 256; i++)
    fprintf(stderr, "debug_image %d: %d\n", i, debug_image[i]);
  */

  //unsigned char *in, float *out, int total
  to_float<<<dimGrid, dimBlock>>>(intImage, deviceOutput, total);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the device memory back to the host here
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutput, total * sizeof(float), cudaMemcpyDeviceToHost));
  wbImage_setData(outputImage, hostOutputImageData);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbSolution(args, outputImage);
  
  //@@ insert code here

  wbTime_start(GPU, "Freeing GPU Memory");
  // Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));
  wbCheck(cudaFree(intImage));
  wbCheck(cudaFree(gray));
  wbCheck(cudaFree(hist));
  wbCheck(cudaFree(cdf));
  wbCheck(cudaFree(mincdf));
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  // Free host memory
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

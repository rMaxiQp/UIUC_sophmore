#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 512
/*
   __global__ void SpMV_JDS_T(int num_rows, float *data, 
	int *col_index, int *jds_t_col_ptr, int *jds_row_index, 
	float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
     float dot = 0;
     unsigned in sec = 0;
      while (jds_t_col_ptr[sec+1]-jds_t_col_ptr[sec] > row){
        dot += data[jds_t_col_ptr[sec]+row] * x[col_index[jds_t_col_ptr[sec]+row]];
        sec++;
        }
      y[jds_row_index[row]] = dot;
      }
    }
 */

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for jds format
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < dim) {
    float dot = 0.0;
    int end = matRows[idx];
    for(int sec = 0; sec < end; sec++) {
      dot += matData[matColStart[sec] + idx] * vec[matCols[matColStart[sec] + idx]];
    }
    out[matRowPerm[idx]] = dot;
  }
}

/*
  spmvJDS(
  deviceOutput => out
  deviceJDSColStart => matColStart
  deviceJDSCols => matCols
  deviceJDSRowPerm => matRowPerm
  deviceJDSRows => matRows
  deviceJDSData => matData
  deviceVector => vec 
  dim => dim);
*/
static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for jds format
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((dim - 1) / BLOCK_SIZE + 1, 1, 1);
  spmvJDSKernel<<<dimGrid, dimBlock>>>(out, matColStart, matCols, matRowPerm, 
                                       matRows, matData, vec, dim);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

  hostOutput = (float *)malloc(sizeof(float) * dim);

  wbTime_stop(Generic, "Importing data and creating memory on host");
  /*
  CSRToJDS(int dim, int *csrRowPtr, int *csrColIdx, float *csrData, int **jdsRowPerm, 
           int **jdsRowNNZ, int **jdsColStartIdx, int **jdsColIdx, float **jdsData);
  */
  CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
           &hostJDSColStart, &hostJDSCols, &hostJDSData);
  maxRowNNZ = hostJDSRows[0];

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
  cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
  cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
          deviceJDSData, deviceVector, dim);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  cudaFree(deviceJDSColStart);
  cudaFree(deviceJDSCols);
  cudaFree(deviceJDSRowPerm);
  cudaFree(deviceJDSRows);
  cudaFree(deviceJDSData);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  free(hostJDSColStart);
  free(hostJDSCols);
  free(hostJDSRowPerm);
  free(hostJDSRows);
  free(hostJDSData);

  return 0;
}


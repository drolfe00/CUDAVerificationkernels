#include <stdio.h>

 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
   
  if (my_index < num_elements) {

    if (my_index%2 == 1) {
       
      input_array[my_index] = my_index;
    } else {
       
      input_array[my_index] = input_array[my_index+1];
    }
  }
}

int main(void) {
   
  int num_elements = 100;
  int num_bytes = sizeof(unsigned int) * num_elements;
    
   
  unsigned int *host_array = 0;
  unsigned int *device_array = 0;
 
   
  host_array = (unsigned int*) malloc(num_bytes);
  cudaMalloc((void **) &device_array, num_bytes);

   
  if (host_array == 0) {
    printf("Unable to allocate memory on host");
    return 1;
  }

  if (device_array == 0) {
    printf("Unable to allocate memory on device");
    return 1;
  }

   
  for (int i = 0; i<num_elements; i++) {
    host_array[i] = 1;
  }

   
  cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

   
  int block_size = 128;
  int grid_size = (num_elements + block_size - 1) / block_size;
{ __set_CUDAConfig(grid_size, block_size); 
          
 device_global(device_array, num_elements);}
          
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

   
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

   
  for (int i=0; i<num_elements; i++) {
    printf("%03u, ", host_array[i]);
    if (i%10 == 9) {
      printf(" \n");
    }
  }

   
  free(host_array);
  cudaFree(device_array);
}

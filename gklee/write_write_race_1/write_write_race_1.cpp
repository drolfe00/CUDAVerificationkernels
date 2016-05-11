#include <stdio.h>

__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  input_array[0] = my_index;
  }


int main(void) {
   
  int num_elements = 1;
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
    host_array[i] = 0;
  }

   
  cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

   
  int block_size = 32;
  int grid_size = (num_elements + block_size - 1) / block_size;
{ __set_CUDAConfig(grid_size, block_size); 
          
 device_global(device_array, num_elements);}
          

   
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

   
  printf("host_array[0] = %u \n", host_array[0]);

   
  free(host_array);
  cudaFree(device_array);
}
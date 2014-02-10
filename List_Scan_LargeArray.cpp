// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan(float * input, float * output, int len ,int bx) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

	__shared__ float XY[2*BLOCK_SIZE] ; 
	unsigned int tx = threadIdx.x ;
//	unsigned int bx = bx ;
	unsigned int start = 2 * bx * blockDim.x ; 
	
	
	if(start + tx < len ) 
		XY[tx] = input[start + tx ] ; 
	else 
		XY[tx] = 0.0;

	if(start + tx + blockDim.x < len )
		XY[tx + blockDim.x] = input[start + tx + blockDim.x] ; 
	else 
		XY[tx + blockDim.x] = 0.0;
	
	__syncthreads();

	for(unsigned int stride = 1 ; stride <= BLOCK_SIZE ; stride *= 2)
	{
		unsigned int index = (tx + 1 ) * stride * 2 - 1 ; 
		if(index < 2 * BLOCK_SIZE) 
			XY[index] += XY[index - stride ] ;
		__syncthreads();
	}

	__syncthreads();

	for(unsigned int stride = BLOCK_SIZE / 2 ; stride > 0 ; stride /= 2)
	{
		unsigned int index = (tx + 1 ) * stride * 2 - 1 ; 
		if(index + stride < 2 * BLOCK_SIZE) 
			XY[index + stride] += XY[index] ; 
		__syncthreads();
	}

	__syncthreads();
	
	if(start + tx < len ) 
		output[start + tx ] += XY[tx]; 

	if(start + tx + blockDim.x < len )
		output[start + tx + blockDim.x ] += XY[tx + blockDim.x];

	__syncthreads();
	
	float in1 = output[start + 2 * BLOCK_SIZE - 1] ;

	int nst = (bx + 1 )* BLOCK_SIZE * 2 ;

	if(nst + tx  < len )
		output[ tx + nst ] += in1  ;

	if(nst + tx + BLOCK_SIZE < len )
		output[nst + tx + blockDim.x ] += in1 ;

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	
	dim3 dimBlock(BLOCK_SIZE , 1 ,1) ; 


    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	
	for(int i = 0 ; i <(numElements - 1 ) /(2 * BLOCK_SIZE) + 1 ; i ++ )
		scan<<<1 , dimBlock>>>(deviceInput , deviceOutput , numElements ,i ) ;

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");
	


    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


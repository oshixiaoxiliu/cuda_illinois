// MP 1
#include	<wb.h>

#define STREAM_NUM 4
#define SEGSIZE 16384


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	if(i < len )
		out[i] = in1[i] + in2[i] ; 
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    //float * deviceInput1;
	//float * deviceInput2;
	float * d_A[STREAM_NUM] ; 
	float * d_B[STREAM_NUM] ; 
	float * d_C[STREAM_NUM] ; 
   // float * deviceOutput;
	cudaStream_t stream[STREAM_NUM] ; 
	
	int nbytes =  SEGSIZE*sizeof(float);

	for(int i = 0 ; i < STREAM_NUM ; i ++ )
	{
		cudaStreamCreate(&stream[i]);
		cudaMalloc((void**)&d_A[i] , nbytes ) ;
		cudaMalloc((void**)&d_B[i] , nbytes ) ;
		cudaMalloc((void**)&d_C[i] , nbytes ) ;
	}

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	
	cudaHostRegister(&hostOutput , inputLength * sizeof(float) , cudaHostRegisterPortable );
	
    wbTime_start(Compute, "Performing CUDA computation");

	for(int i = 0 ; i < inputLength; i += SEGSIZE * STREAM_NUM)
	{
		for(int j = 0 ; j < STREAM_NUM ; j ++ )
			cudaMemcpyAsync(d_A[j] , hostInput1 + i + j * SEGSIZE , nbytes , cudaMemcpyHostToDevice , stream[j] );
		
		for(int j = 0 ; j < STREAM_NUM ; j ++ )
			cudaMemcpyAsync(d_B[j] , hostInput2 + i + j * SEGSIZE , nbytes , cudaMemcpyHostToDevice , stream[j] );

		for(int j = 0 ; j < STREAM_NUM ; j ++ )
			vecAdd<<<SEGSIZE / 256 , 256 , 0 , stream[j] >>>(d_A[j] , d_B[j] , d_C[j] , SEGSIZE );

		for(int j = 0 ; j < STREAM_NUM ; j ++ )
			cudaMemcpyAsync(hostOutput + i + j * SEGSIZE , d_C[j] , nbytes , cudaMemcpyDeviceToHost , stream[j] );
	}
	
	cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation");


	for(int i = 0 ; i < STREAM_NUM ; i ++ )
	{
		cudaStreamDestroy(stream[i]);
		cudaFree( d_A[i] ) ;
		cudaFree( d_B[i] ) ;
		cudaFree( d_C[i] ) ;
	}

    wbSolution(args, hostOutput, inputLength);
	
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


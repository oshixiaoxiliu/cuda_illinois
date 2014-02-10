#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_radius MASK_WIDTH/2

#define O_TILE_WIDTH 28
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1) 

#define CHANNELS 3

//@@ INSERT CODE HERE

__global__ void Convolution_2D_Kernel(float *P , float *N , int height , int width ,
										int channels , const float * __restrict__ M )
{
	int tx = threadIdx.x , ty = threadIdx.y ;
	int bx = blockIdx.x , by = blockIdx.y ; 
	int row_o = by * O_TILE_WIDTH + ty ; 
	int col_o = bx * O_TILE_WIDTH + tx ; 
	int row_i = row_o - MASK_radius ; 
	int col_i = col_o - MASK_radius ; 
	
	__shared__ float NS[BLOCK_WIDTH][BLOCK_WIDTH * CHANNELS] ; 
	
	if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width )
	{
		for(int k = 0 ; k < channels; k ++ )
			NS[ty][tx * channels + k] = P[(row_i*width + col_i) * channels + k ] ; 
	}
	else 
		for(int k = 0 ; k < channels ; k ++ )
			NS[ty][tx * channels + k] = 0.0 ; 

	__syncthreads();
	
	float op = 0.0 ; 
	if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
	{
		for(int k = 0 ; k < channels ; k ++ ) 
		{
			op = 0.0;
			for(int i = 0 ; i < MASK_WIDTH ; i ++ )
				for(int j = 0 ; j < MASK_WIDTH ; j ++ )
				{
					op += M[i * MASK_WIDTH + j] * NS[i+ty][(j+tx)*channels + k ];
				}
			if(row_o < height && col_o < width)
				N[(row_o * width + col_o)*channels + k ] = op ; 
		}
	}
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 dimBlock(BLOCK_WIDTH , BLOCK_WIDTH , 1 ) ; 
	dim3 dimGrid( (imageWidth - 1) / O_TILE_WIDTH  + 1 , (imageHeight - 1) / O_TILE_WIDTH + 1 , 1 );

	Convolution_2D_Kernel<<<dimGrid , dimBlock>>>(deviceInputImageData , deviceOutputImageData ,
												imageHeight ,  imageWidth ,imageChannels  , deviceMaskData );

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

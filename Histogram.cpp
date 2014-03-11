// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void kernelCastFromFloatToUchar(float *in , unsigned char *out , int len )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ; 
    if(i < len) 
        out[i] = (unsigned char )(255 * in[i]);
}
__global__ void kernelConvertFromRGBToGrayScale(unsigned char * in , unsigned char *out , int width , int height )
{
    int col = blockIdx.x * blockDim.x + threadIdx.x ; // width    jj 
    int row = blockIdx.y * blockDim.y + threadIdx.y ; // height   ii

    if(col >= width || row >= height ) return ; 

    int idx = row * width + col ; 
    float r , g , b ;
    r = (float)(in[idx * 3 ] );
    g = (float)(in[idx * 3 + 1 ] ); 
    b = (float)(in[idx * 3 + 2 ] );
    out[idx] = (unsigned char )(0.21 * r + 0.71 * g + 0.07 * b );
}

__global__ void kernelComputeHistogramOfGrayImage(unsigned char * buffer  , unsigned int * histogram  , int size )
{
    __shared__ unsigned int histo_private[256];
  //  if(threadIdx.x < 256 ) 
        histo_private[threadIdx.x] = 0 ; 
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int stride = blockDim.x * gridDim.x ; 
    while(i < size )
    {
        atomicAdd(&(histo_private[buffer[i]]) , 1 );
        i += stride ; 
    }
    __syncthreads();

   //if(threadIdx.x < 256 ) 
   // {
        atomicAdd(&(histogram[threadIdx.x]) , histo_private[threadIdx.x]);
   // }
}

int imageWidth;
int imageHeight;
int imageChannels;

float p(int x ) 
{
    float xx = x ; 
    float yy = imageHeight * imageWidth ; 
    return xx / yy ;
}
unsigned char clamp(unsigned char x , unsigned char  start , unsigned char end)
{
    return (unsigned char)min(max((unsigned int)(x) , (unsigned int)(start)) , (unsigned int)(end) ) ;
}


    float * cdf; 
    float cdfmin;
unsigned char correct_color(unsigned char val) 
{
    float xx = (cdf[(unsigned int)(val)] - cdfmin ) / (1 - cdfmin ) ; 
    unsigned char x = (unsigned char ) (255 * xx );  
    return clamp((unsigned int)(x) , 0 , 255 );
}

int main(int argc, char ** argv) {
    wbArg_t args;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	
	unsigned char * ucharImage;

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
	int size = imageWidth * imageHeight * imageChannels ;

    // getDataOfImage
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	ucharImage = (unsigned char * ) malloc(sizeof(unsigned char) * size );
	

    // Implement a kernel that casts the image from float * to unsigned char *.
 	float * deviceInputImageData ;
    unsigned char * deviceOutputImageData ; 
	cudaMalloc((void**)&deviceInputImageData , sizeof(float)* size );
    cudaMalloc((void**)&deviceOutputImageData , sizeof(unsigned char ) * size);
	cudaMemcpy(deviceInputImageData , hostInputImageData , sizeof(float) * size,cudaMemcpyHostToDevice ) ;
	kernelCastFromFloatToUchar<<< (size - 1 ) / 256 + 1 ,  256 >>>(deviceInputImageData , deviceOutputImageData , size);
	cudaMemcpy(ucharImage , deviceOutputImageData , sizeof(unsigned char)*size , cudaMemcpyDeviceToHost);
    
    //cudaFree(deviceOutputImageData) ; 
    cudaFree(deviceInputImageData);


    // Convert the image from RGB to GrayScale
    unsigned char * hostGrayImage  ;
    unsigned char * deviceGrayImage ; 
    hostGrayImage = (unsigned char *)malloc(sizeof(unsigned char) * imageWidth * imageHeight ) ;
    cudaMalloc((void**)&deviceGrayImage ,sizeof(unsigned char) * imageHeight * imageWidth ); 
    cudaMemcpy(deviceOutputImageData ,  ucharImage , sizeof(unsigned char) * size , cudaMemcpyHostToDevice );
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((imageWidth - 1 ) / 16 + 1 , (imageHeight - 1 ) / 16 + 1 , 1 );  
    kernelConvertFromRGBToGrayScale<<<dimGrid , dimBlock >>>(deviceOutputImageData , deviceGrayImage , imageWidth , imageHeight);
    cudaMemcpy(hostGrayImage , deviceGrayImage , sizeof(unsigned char)* imageWidth * imageHeight , cudaMemcpyDeviceToHost );

    cudaFree(deviceOutputImageData);

    // Implement a kernel that computes the histogram (like in the lectures) of the image.
    int ssize = imageWidth * imageHeight ; 
    unsigned int *deviceHistogram;
    unsigned int *hostHistogram ;
    hostHistogram = (unsigned int * ) malloc(sizeof(unsigned int ) * 256) ;    
    cudaMemcpy(deviceGrayImage , hostGrayImage , sizeof(unsigned char)* imageWidth * imageHeight , cudaMemcpyHostToDevice );
    cudaMalloc((void**)&deviceHistogram , sizeof(unsigned int ) * 256 );
    kernelComputeHistogramOfGrayImage<<<16 , 256>>>(deviceGrayImage , deviceHistogram , ssize);
    cudaMemcpy(hostHistogram , deviceHistogram , sizeof(unsigned int) * 256 , cudaMemcpyDeviceToHost);

    cudaFree(deviceHistogram);
    cudaFree(deviceGrayImage);
    free(hostGrayImage);


    //Compute the Comulative Distribution Function of histogram

    cdf = (float * ) malloc(sizeof(float) * 257 ); 
    cdf[0] = p(hostHistogram[0]);
    for(int i = 1 ; i <= 256 ; i ++ )
    {
        cdf[i] = cdf[i-1] + p(hostHistogram[i]);
    }
    cdfmin = cdf[0] ; 
    for(int i =1 ; i <= 256 ; i++ )
    {
        cdfmin = min(cdfmin , cdf[i]);
    }


    //Apply the histogram equalization function

    for(int i = 0 ; i < size ; i ++ )
    {
        ucharImage[i] = correct_color(ucharImage[i]);
    }

    // Cast back to float
    for(int i = 0 ; i < size ; i ++ )
    {
        hostOutputImageData[i] = (float)(((float)(ucharImage[i])) / 255.0);
    }

    free(cdf);
    free(ucharImage);
    free(hostHistogram);
    wbSolution(args, outputImage);

    //@@ insert code here

    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}

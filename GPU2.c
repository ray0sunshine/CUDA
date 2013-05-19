#include "reference_calc.cpp"
#include "utils.h"

#define blockDimension 32

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

__global__
void gaussian_blur(	const unsigned char * const d_in,
					unsigned char * const d_out,
					int rows,
					int cols,
					const float * const filter,
					const int filterWidth){
	int x = (blockIdx.x * blockDimension) + threadIdx.x;
	int y = (blockIdx.y * blockDimension) + threadIdx.y;
	
	if(x < cols-1 && y < rows-1 && x > 1 && y > 1){
		i = y * cols + x;
		int r = d_in[i-cols-1]+
				d_in[i-cols]+
				d_in[i-cols+1]+
				d_in[i-1]+
				d_in[i]+
				d_in[i+1]+
				d_in[i+cols-1]+
				d_in[i+cols]+
				d_in[i+cols+1];
		d_out[i] = r/9;
	}
	
	/*
	int filterRad = filterWidth/2;
	int i,j,k,l,m,p,q;
	float result = 0.0;
	if(x < cols && y < rows){
		i = y * cols + x;
		q = 0;
		for(j = -filterRad; j <= filterRad; j++){
			l = (y + j)*cols;
			p = (j + filterRad)*filterWidth;
			for(k = -filterRad; k = filterRad; k++){
				m = x + k;
				if(l >= 0 && (y + j) < rows && m >= 0 && m < cols){
					result += filter[p + k + filterRad] * (float)d_in[l + m];
					q++;
				}
			}
		}
		result = (result * filterWidth * filterWidth)/q + 0.5;//round result val, re-average to account for out of range filter pixels
		d_out[i] = (int)(result);
	}*/
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDimension + threadIdx.x, blockIdx.y * blockDimension + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

__global__
void separateChannels(	const uchar4 * const d_in,
						unsigned char * const d_red,
						unsigned char * const d_green,
						unsigned char * const d_blue,
						int rows,
						int cols){
	int x = (blockIdx.x * blockDimension) + threadIdx.x;
	int y = (blockIdx.y * blockDimension) + threadIdx.y;
	int i;
	if(x < cols && y < rows){
		i = y * cols + x;
		d_red[i] = d_in[i].x;
		d_green[i] = d_in[i].y;
		d_blue[i] = d_in[i].z;
	}
}

__global__
void testKern(	uchar4 * const d_in,
				uchar4 * const d_out,
				const size_t rows,
				const size_t cols){
	int x = (blockIdx.x * blockDimension) + threadIdx.x;
	int y = (blockIdx.y * blockDimension) + threadIdx.y;
	int i;
	if(x < cols && y < rows){
		i = y * cols + x;
		d_out[i] = d_in[i];
	}
}

void allocateMemoryAndCopyToGPU(const size_t rows,
								const size_t cols,
                                const float* const h_filter,
								const size_t filterWidth){
	int pixels = rows * cols;
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * pixels));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * pixels));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * pixels));
	checkCudaErrors(cudaMalloc(&d_filter,sizeof(float) * filterWidth * filterWidth));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

//just get the finished uchar4 rgb values into d_out, assume they'll cudaMemcpy it out
void your_gaussian_blur(const uchar4 * const h_in,
						uchar4 * const d_in,
						uchar4 * const d_out,
						const size_t rows,
						const size_t cols,
						unsigned char * d_rBlur,
						unsigned char * d_gBlur,
						unsigned char * d_bBlur,
						const int filterWidth){
	//1024 thread blocks
	const dim3 blockSize(blockDimension, blockDimension);
	
	//use blocks to fill up the image
	const dim3 gridSize(1+((cols-1)/blockDimension), 1+((rows-1)/blockDimension));
	
	//testKern<<<gridSize,blockSize>>>(d_in, d_out, rows, cols);
	
	separateChannels<<<gridSize,blockSize>>>(d_in, d_red, d_green, d_blue, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	gaussian_blur<<<gridSize,blockSize>>>(d_red, d_rBlur, rows, cols, d_filter, filterWidth);
	gaussian_blur<<<gridSize,blockSize>>>(d_green, d_gBlur, rows, cols, d_filter, filterWidth);
	gaussian_blur<<<gridSize,blockSize>>>(d_blue, d_bBlur, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	recombineChannels<<<gridSize,blockSize>>>(d_rBlur, d_gBlur, d_bBlur, d_out, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}

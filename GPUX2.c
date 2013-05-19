#include "reference_calc.cpp"
#include "utils.h"

#define blockDimension 32
#define blockMax 81
__constant__ float const_filter[blockMax];

__global__
void gaussian_blur(const uchar4* const input,
                   uchar4* const output,
                   int numRows,
				   int numCols,
				   const int filterWidth)
{
    int x = (blockIdx.x * blockDimension) + threadIdx.x;
	int y = (blockIdx.y * blockDimension) + threadIdx.y;
	int filterRad = filterWidth/2;
	int i,j,k,l,m,p;
	float r,g,b;
	if(x < numCols && y < numRows){
		i = y * numCols + x;
		r = 0;
		g = 0;
		b = 0;
		for(j = -filterRad; j <= filterRad; j++){
			l = (y + j)*numCols;
			p = (j + filterRad)*filterWidth;
			for(k = -filterRad + ((j+filterRad)%2); k <= filterRad; k+=2){
				m = x + k;
				if(l < 0){
					l = 0;
				}else if((y + j) >= numRows){
					l = (numRows - 1)*numCols;
				}
				if(m < 0){
					m = 0;
				}else if(m >= numCols){
					m = numCols - 1;
				}
				r += const_filter[p + k + filterRad] * input[l + m].x;
				g += const_filter[p + k + filterRad] * input[l + m].y;
				b += const_filter[p + k + filterRad] * input[l + m].z;
			}
		}
		output[i] = make_uchar4(r*2, g*2, b*2, 255);
	}
}

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
								const size_t numColsImage,
                                const float* const h_filter,
								const size_t filterWidth)
{
	cudaMemcpyToSymbol(const_filter, h_filter, sizeof(float) * filterWidth * filterWidth, 0, cudaMemcpyHostToDevice);
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA,
						uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
						const size_t numRows,
						const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(blockDimension, blockDimension);
  const dim3 gridSize(1+((numCols-1)/blockDimension), 1+((numRows-1)/blockDimension));
  gaussian_blur<<<gridSize,blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, filterWidth);
}

void cleanup(){}
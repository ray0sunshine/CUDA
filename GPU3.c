/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "reference_calc.cpp"
#include "utils.h"

#define fullBlock 1024

__global__
void getMin(float* d_lum, float* d_min){
	int i = 1 + 2*((blockIdx.x * fullBlock) + threadIdx.x);
	int step = 1;
	while(step <= fullBlock){
		if((i+1)%(step*2) == 0){
			if(d_lum[i] > d_lum[i-step]){
				d_lum[i] = d_lum[i-step];
			}
			step *= 2;
			if(step > fullBlock && d_lum[i] < *d_min){
				*d_min = d_lum[i];
			}
		}
		__syncthreads();
	}
}

__global__
void getMax(float* d_lum, float* d_max){
	int i = 1 + 2*((blockIdx.x * fullBlock) + threadIdx.x);
	int step = 1;
	while(step <= fullBlock){
		if((i+1)%(step*2) == 0){
			if(d_lum[i] < d_lum[i-step]){
				d_lum[i] = d_lum[i-step];
			}
			step *= 2;
			if(step > fullBlock && d_lum[i] > *d_max){
				*d_max = d_lum[i];
			}
		}
		__syncthreads();
	}
}

__global__
void hashToBin(const float* const d_lum, unsigned int* const d_cdf, const float range, const size_t bins, int pixCount, float minlum){
	int i = (blockIdx.x * fullBlock) + threadIdx.x;
	if(i < pixCount){
		int idx = ((d_lum[i] - minlum)/range)*bins;
		if(idx == bins){
			idx -= 1;
		}
		atomicAdd(&(d_cdf[idx]),1);
	}
}

__global__
void makeCdf(const int halfBins, unsigned int* d_cdf){
	int i = 1 + 2*threadIdx.x;
	int step = 1;
	int swap;
	while(step <= halfBins){
		if((i+1)%(step*2) == 0){
			d_cdf[i] += d_cdf[i-step];
		}
		step *= 2;
		__syncthreads();
	}
	
	if(i == (halfBins*2)-1){
		d_cdf[i] = 0;
	}
	
	while(step >= 2){
		if((i+1)%step == 0){
			swap = d_cdf[i];
			d_cdf[i] += d_cdf[i-(step/2)];
			d_cdf[i-(step/2)] = swap;
		}
		step /= 2;
		__syncthreads();
	}
}

void your_histogram_and_prefixsum(const float* const d_lum,
                                  unsigned int* const d_cdf,
                                  float &minlum,
                                  float &maxlum,
                                  const size_t rows,
                                  const size_t cols,
                                  const size_t bins)
{
	int pixCount = rows * cols;
	int blocks = 1+(((pixCount)-1)/fullBlock);

	minlum = 9001;
	maxlum = -9001;
	
	//minlum = -20;
	//maxlum = 2.43933269383;
	
	/*float *d_min;
	float *d_max;
	float *d_lum2;
	float *d_lum3;
	
	cudaMalloc(&d_min, sizeof(float));
	cudaMalloc(&d_max, sizeof(float));
	cudaMalloc(&d_lum2, pixCount*sizeof(float));
	cudaMalloc(&d_lum3, pixCount*sizeof(float));
	
	cudaMemcpy(d_min, &minlum, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, &maxlum, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lum2, d_lum, pixCount*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_lum3, d_lum, pixCount*sizeof(float), cudaMemcpyDeviceToDevice);
	
	int fullBL = pixCount/(fullBlock*2);
	int partBL = pixCount%(fullBlock*2);*/
	
	/*
	getMin<<<fullBL,fullBlock>>>(d_lum2, d_min);
	getMax<<<fullBL,fullBlock>>>(d_lum3, d_max);

	cudaMemcpy(&minlum, d_min, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxlum, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	*/
	
	float *h_lum;
	h_lum = (float*)malloc(pixCount*sizeof(float));
	cudaMemcpy(h_lum, d_lum, pixCount*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<pixCount; i++){
		if(h_lum[i] < minlum){
			minlum = h_lum[i];
		}else if(h_lum[i] > maxlum){
			maxlum = h_lum[i];
		}
	}
	
	float h_range = maxlum - minlum;
	hashToBin<<<blocks,fullBlock>>>(d_lum, d_cdf, h_range, bins, pixCount, minlum);
	
	int halfBins = bins/2;
	
	makeCdf<<<1,halfBins>>>(halfBins, d_cdf);
	
	free(h_lum);
	/*cudaFree(d_min);
	cudaFree(d_max);
	cudaFree(d_lum2);
	cudaFree(d_lum3);*/
	
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = ((lum[i] - lumMin) / lumRange) * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}
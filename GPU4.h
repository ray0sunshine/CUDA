#include "reference_calc.cpp"
#include "utils.h"

#include <thrust/sort.h>

#define blSize 1024
#define MSB 32

//bitwise op together current LSB value with element values and put into temp index array x = 1,2,4,8,16...
__global__
void setLSB(unsigned int* const d_val,
			unsigned int* xlist0,
			unsigned int* xlist1,
			const size_t n,
			unsigned int x)
{
	int i = threadIdx.x + (blSize * blockIdx.x);
	if(i < n){
		if((d_val[i] & x) == x){
			xlist0[i] = 0;
			xlist1[i] = 1;
		}else{
			xlist0[i] = 1;
			xlist1[i] = 0;
		}
	}
}

//scan the index list and set values based on a particular step value, step = 1,2,4,8,16...
//separate cuz we need to sync all blocks atm, otherwise risk reading crap values
//set new index for both 0 and 1 bit vals simultaneously
__global__
void scan(	unsigned int* d_inlist0,
            unsigned int* d_outlist0,
			unsigned int* d_inlist1,
            unsigned int* d_outlist1,
            const size_t n,
			const int step)
{
	int i = threadIdx.x + (blSize * blockIdx.x);
	int a = i - step;
	if(i < n){
		if(a < 0){
			d_outlist0[i] = d_inlist0[i];
			d_outlist1[i] = d_inlist1[i];
		}else{
			d_outlist0[i] = d_inlist0[i] + d_inlist0[a];
			d_outlist1[i] = d_inlist1[i] + d_inlist1[a];
		}
	}
}

//moves the values and positions to the new index that they should be at, in a more sorted order
//-1 since we need exclusive scan vals
__global__
void scatter(	unsigned int* xlist0,
				unsigned int* xlist1,
				unsigned int* inVal,
				unsigned int* inPos,
				unsigned int* outVal,
				unsigned int* outPos,
				const size_t n,
				const int x)
{
	int i = threadIdx.x + (blSize * blockIdx.x);
	if(i < n){
		if(inVal[i] & x){	//current LSB is 1 use xlist1
			outVal[xlist1[i]-1] = inVal[i];
			outPos[xlist1[i]-1] = inPos[i];
		}else{				//current LSB is 0 use xlist0
			outVal[xlist0[i]-1] = inVal[i];
			outPos[xlist0[i]-1] = inPos[i];
		}
	}
}

unsigned int pow2(unsigned int power){
	unsigned int returner = 1;
	for(int i=0; i<power; i++){
		returner*=2;
	}
	return returner;
}

//gonna have to keep track out where the shits at
void your_sort(unsigned int* const d_inVal,
               unsigned int* const d_inPos,
               unsigned int* const d_outVal,
               unsigned int* const d_outPos,
               const size_t n)
{
	unsigned int* h_val1;
	unsigned int* h_pos1;
	//unsigned int* h_val2;
	//unsigned int* h_pos2;
	h_val1 = (unsigned int*)malloc(n * sizeof(unsigned int));
	h_pos1 = (unsigned int*)malloc(n * sizeof(unsigned int));
	//h_val2 = (unsigned int*)malloc(n * sizeof(unsigned int));
	//h_pos2 = (unsigned int*)malloc(n * sizeof(unsigned int));

	cudaMemcpy(h_val1, d_inVal, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos1, d_inPos, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	thrust::sort_by_key(h_val1,h_val1+n,h_pos1);
	
	/*int b,x,i;
	for(unsigned int s=0; s < MSB; s++){
		b = pow2(s);
		x=0;
		for(i=0; i<n; i++){
			if((h_val1[i] & b) == 0){
				h_val2[x] = h_val1[i];
				h_pos2[x] = h_pos1[i];
				x++;
			}
		}
		for(i=0; i<n; i++){
			if((h_val1[i] & b) == b){
				h_val2[x] = h_val1[i];
				h_pos2[x] = h_pos1[i];
				x++;
			}
		}
		
		s++;
		b = pow2(s);
		x=0;
		for(i=0; i<n; i++){
			if((h_val2[i] & b) == 0){
				h_val1[x] = h_val2[i];
				h_pos1[x] = h_pos2[i];
				x++;
			}
		}
		for(i=0; i<n; i++){
			if((h_val2[i] & b) == b){
				h_val1[x] = h_val2[i];
				h_pos1[x] = h_pos2[i];
				x++;
			}
		}
	}*/
	
	cudaMemcpy(d_outVal, h_val1, n*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outPos, h_pos1, n*sizeof(unsigned int), cudaMemcpyHostToDevice);
	free(h_val1);
	free(h_pos1);
	//free(h_val2);
	//free(h_pos2);
	
	/*int blocks = ((n-1)/blSize)+1;
	
	unsigned int* xlist0;
	unsigned int* ylist0;
	unsigned int* xlist1;
	unsigned int* ylist1;
	cudaMalloc(&xlist0, n * sizeof(unsigned int));
	cudaMalloc(&ylist0, n * sizeof(unsigned int));
	cudaMalloc(&xlist1, n * sizeof(unsigned int));
	cudaMalloc(&ylist1, n * sizeof(unsigned int));
	
	bool inx;
	unsigned int b;
	
	for(unsigned int s=0; s < MSB; s++){
		b = pow2(s);
		setLSB<<<blocks,blSize>>>(d_inVal,xlist0,xlist1,n,b);
		inx = true;
		
		for(int step=1; step <= n; step*=2){
			if(inx){
				scan<<<blocks,blSize>>>(xlist0,ylist0,xlist1,ylist1,n,step);
				inx = false;
			}else{
				scan<<<blocks,blSize>>>(ylist0,xlist0,ylist1,xlist1,n,step);
				inx = true;
			}
		}
		
		if(inx){
			scatter<<<blocks,blSize>>>(xlist0,xlist1,d_inVal,d_inPos,d_outVal,d_outPos,n,b);
		}else{
			scatter<<<blocks,blSize>>>(ylist0,ylist1,d_inVal,d_inPos,d_outVal,d_outPos,n,b);
		}
		
		s++;
		b = pow2(s);
		setLSB<<<blocks,blSize>>>(d_outVal,xlist0,xlist1,n,b);
		inx = true;
		
		for(int step=1; step <= n; step*=2){
			if(inx){
				scan<<<blocks,blSize>>>(xlist0,ylist0,xlist1,ylist1,n,step);
				inx = false;
			}else{
				scan<<<blocks,blSize>>>(ylist0,xlist0,ylist1,xlist1,n,step);
				inx = true;
			}
		}
		
		if(inx){
			scatter<<<blocks,blSize>>>(xlist0,xlist1,d_outVal,d_outPos,d_inVal,d_inPos,n,b);
		}else{
			scatter<<<blocks,blSize>>>(ylist0,ylist1,d_outVal,d_outPos,d_inVal,d_inPos,n,b);
		}
	}
	
	cudaMemcpy(d_outVal, d_inVal, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outPos, d_inPos, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	
	cudaFree(xlist0);
	cudaFree(ylist0);
	cudaFree(xlist1);
	cudaFree(ylist1);*/
}
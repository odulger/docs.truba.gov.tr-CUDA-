#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <curand.h>
#include <unistd.h>
#include "RNG_functions.cuh"

using namespace std;

__global__ void full_coalesced_access(float *A,float *B,float *C)
{
    unsigned int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	for(int i=0;i<100;i++)
	{
		C[tid] = A[tid] + B[tid];//Vector addition
	}
}

__global__ void non_coalesced_access(float *A,float *B,float *C,curandInitializer RNGs,unsigned int NP)
{
    unsigned int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id

	curandState_t state;// State of the generator
	RNGs.load(state,tid);// Loading the state
	unsigned int index,i;

	for(i=0;i<100;i++)
	{
		index = curand(&state)/NP;// Generate a random number between 0 and size-1
		C[tid] = A[index] + B[index];//Vector addition
	}
}

__global__ void semi_coalesced_access(float *A,float *B,float *C,curandInitializer RNGs1,curandInitializer RNGs2,unsigned int NP_group_count,unsigned int NP_group_size,unsigned int GS)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id

	curandState_t state1,state2;// States of the generators
	RNGs1.load(state1,tid);// Loading the state of first generator
	RNGs2.load(state2,tid);// Loading the state of second generator

	unsigned int GN = curand(&state2)/NP_group_count;//Generate a random number between 0 and group_count-1
	unsigned int index,i;

	for(i=0;i<100;i++)
	{
		index = (curand(&state1)/NP_group_size) + (GN*GS);//Generate a random number between 0 and group_size-1 then shift the index
		C[tid] = A[index] + B[index];//Vector addition
	}
}

int main(int argc, char **argv)
{
	unsigned int data_size = 4194304;//Data size
	float *A_host = new float[data_size];//Host Array
	float *B_host = new float[data_size];//Host Array
	float *C_host = new float[data_size];//Host Array
	
	float *A_GPU,*B_GPU,*C_GPU;//Device Arrays
	cudaMalloc((void**)&A_GPU,sizeof(float)*data_size);
	cudaMalloc((void**)&B_GPU,sizeof(float)*data_size);
	cudaMalloc((void**)&C_GPU,sizeof(float)*data_size);

	for(int counter = 0;counter < data_size; counter++)
	{
		A_host[counter] = counter+1;//Assigning numbers from 1 to size
		B_host[counter] = counter+2;//Assigning numbers from 2 to size+1 
	}

	cudaMemcpy(A_GPU,A_host,sizeof(float)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU,B_host,sizeof(float)*data_size,cudaMemcpyHostToDevice);

	unsigned int WS = 32;//Warp size
	unsigned int NTB = 1024;//Number of threads in a block
	unsigned int NP_data_size = (unsigned long int)pow(2,32)/data_size;// Number of partitions in a period for 'data_size'

	unsigned int segment_size = 128;//Number of bytes of a segment
	unsigned int group_size = 16*(segment_size/4);//Number of data in a group	
	unsigned int group_count = data_size/group_size;//Number of groups for 'data_size'
	unsigned int NP_group_count = (unsigned long int)pow(2,32)/group_count;// Number of partitions in a period for 'group_count'
	unsigned int NP_group_size = (unsigned long int)pow(2,32)/group_size;// Number of partitions in a period for 'group_size'
	
	dim3 threadsPerBlock(NTB);//Number of threads in a block
	dim3 numBlocks(data_size/NTB);//Number of blocks in a grid

	curandInitializer RNGs(data_size);
	unsigned int clck = clock();
	initialize_RNGs<<<numBlocks,threadsPerBlock>>>(RNGs,clck);//Creating a generator for 'non_coalesced_access' kernel

	curandInitializer RNGs1(data_size);
	clck = clock();
	initialize_RNGs<<<numBlocks,threadsPerBlock>>>(RNGs1,clck);//Creating a generator for 'semi_coalesced_access' kernel (to pick elements within a group)

	curandInitializer RNGs2(data_size);
	clck = clock();
	initialize_RNGs<<<numBlocks,threadsPerBlock>>>(RNGs2,clck,WS);//Creating a generator for 'semi_coalesced_access' kernel (to pick a group for a warp)
	

	cudaEvent_t start, stop;//Variable for timer operations
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);
	full_coalesced_access<<<numBlocks,threadsPerBlock>>>(A_GPU,B_GPU,C_GPU);//Launching 'full_coalesced_access' kernel
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time_full = milliseconds/1000.0f;

	cudaEventRecord(start);
	non_coalesced_access<<<numBlocks,threadsPerBlock>>>(A_GPU,B_GPU,C_GPU,RNGs,NP_data_size);//Launching 'non_coalesced_access' kernel
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time_non = milliseconds/1000.0f;

	cudaEventRecord(start);
	semi_coalesced_access<<<numBlocks,threadsPerBlock>>>(A_GPU,B_GPU,C_GPU,RNGs1,RNGs2,NP_group_count,NP_group_size,group_size);//Launching 'semi_coalesced_access' kernel
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time_semi = milliseconds/1000.0f;

	cudaDeviceSynchronize();//Waits until all kernels complete their run

	cout << "Exec. Time of 'full_coalesced_access' kernel = " << time_full << endl;
	cout << "Exec. Time of 'semi_coalesced_access' kernel = " << time_semi << endl;
	cout << "Speed Up = " << time_semi/time_full << "X" << endl;
	cout << "Exec. Time of 'non_coalesced_access' kernel = " << time_non << endl;
	cout << "Speed Up = " << time_non/time_full << "X" << endl;

	cudaMemcpy(C_host,C_GPU,sizeof(float)*data_size,cudaMemcpyDeviceToHost);
	
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess )
		cout << "CUDA Error: " << cudaGetErrorString(err) << endl;

	delete[] A_host;
	delete[] B_host;
	delete[] C_host;
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
}

#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <curand.h>
#include <unistd.h>

using namespace std;

__global__ void warp_no_divergence(float *A,float *B,float *C)//No branching for the warps in 'if-elseif' structure
{
    unsigned int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id

	for(unsigned int i=0;i<100;i++)
	{
		if( (tid/32) % 4 == 0)
			C[tid] = A[tid] + B[tid];//Vector addition
		else if( (tid/32) % 4 == 1)
			C[tid] = A[tid] - B[tid];//Vector subtraction
		else if( (tid/32) % 4 == 2)
			C[tid] = A[tid] * B[tid];//Vector multiplication
		else if( (tid/32) % 4 == 3)
			C[tid] = A[tid] / B[tid];//Vector division
	}
}

__global__ void warp_divergence(float *A,float *B,float *C)//Four different paths for the warps in 'if-elseif' structure
{
    unsigned int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id

	for(unsigned int i=0;i<100;i++)
	{
		if( tid % 4 == 0)
			C[tid] = A[tid] + B[tid];//Vector addition
		else if( tid % 4 == 1)
			C[tid] = A[tid] - B[tid];//Vector subtraction
		else if( tid % 4 == 2)
			C[tid] = A[tid] * B[tid];//Vector multiplication
		else if( tid % 4 == 3)
			C[tid] = A[tid] / B[tid];//Vector division
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

	unsigned int NTB = 1024;//Number of threads in a block
	dim3 threadsPerBlock(NTB);//Number of threads in a block
	dim3 numBlocks(data_size/NTB);//Number of blocks in a grid
	

	cudaEvent_t start, stop;//Variables for timer operations
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);
	warp_no_divergence<<<numBlocks,threadsPerBlock>>>(A_GPU,B_GPU,C_GPU);//Launching 'warp_no_divergence' kernel
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time1 = milliseconds/1000.0f;

	cudaEventRecord(start);
	warp_divergence<<<numBlocks,threadsPerBlock>>>(A_GPU,B_GPU,C_GPU);//Launching 'warp_divergence' kernel
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time2 = milliseconds/1000.0f;

	cudaDeviceSynchronize();//Waits until all kernels complete their run

	cout << "Exec. Time of 'warp_no_divergence' kernel = " << time1 << endl;
	cout << "Exec. Time of 'warp_divergence' kernel = " << time2 << endl;
	cout << "Speed Up = " << time2/time1 << "X" << endl;

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

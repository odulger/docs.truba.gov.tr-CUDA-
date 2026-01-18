#include <iostream>

using namespace std;

__global__ void vector_add_S1(float *A,float *B,float *C)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	for(int i=0;i<1000000;i++)
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

__global__ void vector_add_S2(float *A,float *B,float *C)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	for(int i=0;i<1000000;i++)
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

int main(int argc, char **argv)
{
	int data_size = 65536;//Data size

	float *A_host = new float[data_size];//Host Array
	float *B_host = new float[data_size];//Host Array
	float *C_host = new float[data_size];//Host Array
	
	float *A_GPU,*B_GPU,*C_GPU;//Device Arrays
	cudaMalloc((void**)&A_GPU,sizeof(float)*data_size);
	cudaMalloc((void**)&B_GPU,sizeof(float)*data_size);
	cudaMalloc((void**)&C_GPU,sizeof(float)*data_size);

	for(int counter = 0;counter < data_size; counter++)
	{
		A_host[counter] = counter+1;//Assigning numbers from 1 to data_size
		B_host[counter] = counter+2;//Assigning numbers from 2 to data_size+1 
	}

	cudaMemcpy(A_GPU,A_host,sizeof(float)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU,B_host,sizeof(float)*data_size,cudaMemcpyHostToDevice);

	
	cudaEvent_t start, stop;//Timer variables
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	
  //Senaryo1
	int NTB;
	if(data_size <= 1024)
		NTB = data_size;
	else 
		NTB = 1024;
	dim3 threadsPerBlockS1(NTB);//Number of threads in a block
	dim3 numBlocksS1(data_size/NTB);//Number of blocks in a grid

	cudaEventRecord(start);
	vector_add_S1<<<numBlocksS1,threadsPerBlockS1>>>(A_GPU,B_GPU,C_GPU);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time1 = milliseconds/1000.0f;
	cout << "Execution time of S1 = " << time1 << " seconds" << endl;

	cudaDeviceSynchronize();//Waits until vector_add_S1 kernel completes its run

	cudaMemcpy(C_host,C_GPU,sizeof(float)*data_size,cudaMemcpyDeviceToHost);
	
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess )
		cout << "CUDA Error: " << cudaGetErrorString(err) << endl;

	//////////////////////////////////////////////////////////////////////////
 
	//Senaryo2
	if(data_size <= 512)
		NTB = 32;
	else if(data_size == 1024)
		NTB = 64;
	else if(data_size == 2048)
		NTB = 128;
	else if(data_size == 4096)
		NTB = 256;
	else if(data_size == 8192)
		NTB = 512;
	else
		NTB = 1024;
	dim3 threadsPerBlockS2(NTB);//Number of threads in a block
	dim3 numBlocksS2(data_size/NTB);//Number of blocks in a grid

	cudaEventRecord(start);
	vector_add_S2<<<numBlocksS2,threadsPerBlockS2>>>(A_GPU,B_GPU,C_GPU);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	float time2 = milliseconds/1000.0f;
	cout << "Execution time of S2 = " << time2 << " seconds" << endl;

	cudaDeviceSynchronize();//Waits until vector_add_S2 kernel completes its run

	cudaMemcpy(C_host,C_GPU,sizeof(float)*data_size,cudaMemcpyDeviceToHost);
	
	err = cudaGetLastError();
	if ( err != cudaSuccess )
		cout << "CUDA Error: " << cudaGetErrorString(err) << endl;

	delete[] A_host;
	delete[] B_host;
	delete[] C_host;
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
}

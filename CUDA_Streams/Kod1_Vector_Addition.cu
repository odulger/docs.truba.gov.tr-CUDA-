#include <stdio.h>

__global__ void vector_addition(int *A,int *B,int *C,int size)//CUDA kernel
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	if(tid<size)
		C[tid] = A[tid] + B[tid];//Vector Addition gerçekleşiyor
}

int main()
{
	int size = 1000002;//Dizi büyüklüğü (3'e tam bölünmeli)
	int ThreadPerBlock = 1024;//Blok büyüklüğü (Donanımın izin verdiği en yüksek değer)
	int BlockPerGrid = (size-1)/ThreadPerBlock+1;//Blok sayısı
 
	int *A_Host,*B_Host,*C_Host;
	A_Host = new int[size];//CPU belleğinde (Heap bölgesi) yer açılıyor
	B_Host = new int[size];//CPU belleğinde (Heap bölgesi) yer açılıyor
	C_Host = new int[size];//CPU belleğinde (Heap bölgesi) yer açılıyor
	
	for(int i=1;i<=size;i++)//Diziye başlangıç değerleri atanıyor
	{
		A_Host[i-1] = i;
		B_Host[i-1] = 0;
	}

	int *A_GPU,*B_GPU,*C_GPU;
	cudaMalloc(&A_GPU,sizeof(int)*size);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&B_GPU,sizeof(int)*size);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&C_GPU,sizeof(int)*size);//GPU ana belleğinde yer açılıyor

	dim3 DimBlock(ThreadPerBlock);//Bir bloktaki thread sayısı
	dim3 DimGrid(BlockPerGrid);//Bir griddeki blok sayısı

	cudaEvent_t start, stop;//Süre değişkenleri
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float totaltime;//Toplam süre değişkeni

	cudaEventRecord(start);//Süre başlatıldı

	cudaMemcpy(A_GPU,A_Host,sizeof(int)*size,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpy(B_GPU,B_Host,sizeof(int)*size,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı

	vector_addition<<<DimGrid,DimBlock>>>(A_GPU,B_GPU,C_GPU,size);//CUDA kernel çalıştırılıyor

	cudaMemcpy(C_Host,C_GPU,sizeof(int)*size,cudaMemcpyDeviceToHost);//GPU'dan CPU'ya veri aktarımı

	cudaEventRecord(stop);//Süre durduruldu
	cudaEventSynchronize(stop);//Event işlemleri bitene kadar program beklemekte
	cudaEventElapsedTime(&totaltime, start, stop);//Geçen süre hesaplanıyor
	printf("Toplam Süre = %f saniye\n",totaltime);
	printf("C[size-1] = %d\n",C_Host[size-1]);

	delete[] A_Host;//Dizi CPU belleğinden siliniyor
	delete[] B_Host;//Dizi CPU belleğinden siliniyor
	delete[] C_Host;//Dizi CPU belleğinden siliniyor

	cudaFree(A_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(B_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(C_GPU);//Dizi GPU belleğinden siliniyor

	cudaError_t err = cudaGetLastError();//GPU'da oluşan son hatayı yakalıyor
	if ( err != cudaSuccess )
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
}

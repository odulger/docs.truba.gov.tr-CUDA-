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
	int BlockPerGrid = ((size/3)-1)/ThreadPerBlock+1;//Blok sayısı
	int *A_Host,*B_Host,*C_Host;

	cudaMallocHost((void**)&A_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	cudaMallocHost((void**)&B_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	cudaMallocHost((void**)&C_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	
	for(int i=1;i<=size;i++)//Diziye başlangıç değerleri atanıyor
	{
		A_Host[i-1] = i;
		B_Host[i-1] = 0;
	}

	int *A1_GPU,*B1_GPU,*C1_GPU;
	int *A2_GPU,*B2_GPU,*C2_GPU;
	int *A3_GPU,*B3_GPU,*C3_GPU;
	cudaMalloc(&A1_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&B1_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&C1_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&A2_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&B2_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&C2_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&A3_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&B3_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor
	cudaMalloc(&C3_GPU,sizeof(int)*size/3);//GPU ana belleğinde yer açılıyor

	dim3 DimBlock(ThreadPerBlock);//Bir bloktaki thread sayısı
	dim3 DimGrid(BlockPerGrid);//Bir griddeki blok sayısı

	cudaEvent_t start, stop;//Süre değişkenleri
	cudaEventCreate(&start);//Event oluşturuluyor
	cudaEventCreate(&stop);//Event oluşturuluyor
	float totaltime;//Toplam süre değişkeni

	cudaStream_t stream[3];
	cudaStreamCreate(&stream[0]);//1.stream yaratılıyor
	cudaStreamCreate(&stream[1]);//2.stream yaratılıyor
	cudaStreamCreate(&stream[2]);//3.stream yaratılıyor

	cudaEventRecord(start);//Süre başlatıldı

	//Parçaların işlemleri CUDA Stream'ler kullanılarak overlap ediliyor
	cudaMemcpyAsync(A1_GPU,A_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[0]);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B1_GPU,B_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[0]);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock,0,stream[0]>>>(A1_GPU,B1_GPU,C1_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+0*(size/3),C1_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[0]);//GPU'dan CPU'ya veri aktarımı

	cudaMemcpyAsync(A2_GPU,A_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[1]);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B2_GPU,B_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[1]);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock,0,stream[1]>>>(A2_GPU,B2_GPU,C2_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+1*(size/3),C2_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[1]);//GPU'dan CPU'ya veri aktarımı

	cudaMemcpyAsync(A3_GPU,A_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[2]);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B3_GPU,B_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[2]);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock,0,stream[2]>>>(A3_GPU,B3_GPU,C3_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+2*(size/3),C3_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[2]);//GPU'dan CPU'ya veri aktarımı
	
	cudaStreamSynchronize(stream[0]);//Stream 1'deki işlemler bitene kadar host bekliyor
	cudaStreamSynchronize(stream[1]);//Stream 2'deki işlemler bitene kadar host bekliyor
	cudaStreamSynchronize(stream[2]);//Stream 3'deki işlemler bitene kadar host bekliyor

	cudaEventRecord(stop);//Süre durduruldu
	cudaEventSynchronize(stop);//Event işlemleri bitene kadar program beklemekte
	cudaEventElapsedTime(&totaltime, start, stop);//Geçen süre hesaplanıyor
	printf("Toplam Süre = %f saniye\n",totaltime);
	printf("C[size-1] = %d\n",C_Host[size-1]);

	cudaFreeHost(A_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(B_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(C_Host);//Dizi CPU belleğinden siliniyor

	cudaFree(A1_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(B1_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(C1_GPU);//Dizi GPU belleğinden siliniyor

	cudaFree(A2_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(B2_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(C2_GPU);//Dizi GPU belleğinden siliniyor

	cudaFree(A3_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(B3_GPU);//Dizi GPU belleğinden siliniyor
	cudaFree(C3_GPU);//Dizi GPU belleğinden siliniyor
	
	cudaEventDestroy(start);//Event yok ediliyor
	cudaEventDestroy(stop);//Event yok ediliyor
	
	cudaStreamDestroy(stream[0]);//Stream 1 yok ediliyor
	cudaStreamDestroy(stream[1]);//Stream 2 yok ediliyor
	cudaStreamDestroy(stream[2]);//Stream 3 yok ediliyor

	cudaError_t err = cudaGetLastError();//GPU'da oluşan son hatayı yakalıyor
	if ( err != cudaSuccess )
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
}

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

	int *A1_GPU,*B1_GPU,*C1_GPU;//GPU1 için kullanılacak
	int *A2_GPU,*B2_GPU,*C2_GPU;//GPU2 için kullanılacak
	int *A3_GPU,*B3_GPU,*C3_GPU;//GPU3 için kullanılacak
	
	cudaSetDevice(0);//İlk GPU aktif oluyor
	cudaMalloc(&A1_GPU,sizeof(int)*size/3);//GPU1 ana belleğinde yer açılıyor
	cudaMalloc(&B1_GPU,sizeof(int)*size/3);//GPU1 ana belleğinde yer açılıyor
	cudaMalloc(&C1_GPU,sizeof(int)*size/3);//GPU1 ana belleğinde yer açılıyor
	cudaSetDevice(1);//İkinci GPU aktif oluyor
	cudaMalloc(&A2_GPU,sizeof(int)*size/3);//GPU2 ana belleğinde yer açılıyor
	cudaMalloc(&B2_GPU,sizeof(int)*size/3);//GPU2 ana belleğinde yer açılıyor
	cudaMalloc(&C2_GPU,sizeof(int)*size/3);//GPU2 ana belleğinde yer açılıyor
	cudaSetDevice(2);//Üçüncü GPU aktif oluyor
	cudaMalloc(&A3_GPU,sizeof(int)*size/3);//GPU3 ana belleğinde yer açılıyor
	cudaMalloc(&B3_GPU,sizeof(int)*size/3);//GPU3 ana belleğinde yer açılıyor
	cudaMalloc(&C3_GPU,sizeof(int)*size/3);//GPU3 ana belleğinde yer açılıyor

	dim3 DimBlock(ThreadPerBlock);//Bir bloktaki thread sayısı
	dim3 DimGrid(BlockPerGrid);//Bir griddeki blok sayısı

	cudaEvent_t start, stop;//Süre değişkenleri
	cudaEventCreate(&start);//Event oluşturuluyor
	cudaEventCreate(&stop);//Event oluşturuluyor
	float totaltime;//Toplam süre değişkeni

	cudaEventRecord(start);//Süre başlatıldı

	cudaSetDevice(0);//İlk GPU aktif oluyor
	cudaMemcpyAsync(A1_GPU,A_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B1_GPU,B_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock>>>(A1_GPU,B1_GPU,C1_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+0*(size/3),C1_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost);//GPU'dan CPU'ya veri aktarımı

	cudaSetDevice(1);//İkinci GPU aktif oluyor
	cudaMemcpyAsync(A2_GPU,A_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B2_GPU,B_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock>>>(A2_GPU,B2_GPU,C2_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+1*(size/3),C2_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost);//GPU'dan CPU'ya veri aktarımı

	cudaSetDevice(2);//Üçüncü GPU aktif oluyor
	cudaMemcpyAsync(A3_GPU,A_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	cudaMemcpyAsync(B3_GPU,B_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice);//CPU'dan GPU'ya veri aktarımı
	vector_addition<<<DimGrid,DimBlock>>>(A3_GPU,B3_GPU,C3_GPU,size/3);//CUDA kernel çalıştırılıyor
	cudaMemcpyAsync(C_Host+2*(size/3),C3_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost);//GPU'dan CPU'ya veri aktarımı
	
	cudaSetDevice(0);//İlk GPU aktif oluyor
	cudaDeviceSynchronize();//GPU1'deki işlemler bitene kadar host bekliyor
	cudaSetDevice(1);//İkinci GPU aktif oluyor
	cudaDeviceSynchronize();//GPU2'deki işlemler bitene kadar host bekliyor
	cudaSetDevice(2);//Üçüncü GPU aktif oluyor
	cudaDeviceSynchronize();//GPU3'teki işlemler bitene kadar host bekliyor

	cudaEventRecord(stop);//Süre durduruldu
	cudaEventSynchronize(stop);//Event işlemleri bitene kadar program beklemekte
	cudaEventElapsedTime(&totaltime, start, stop);//Geçen süre hesaplanıyor
	printf("Toplam Süre = %f saniye\n",totaltime);
	printf("C[size-1] = %d\n",C_Host[size-1]);

	cudaFreeHost(A_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(B_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(C_Host);//Dizi CPU belleğinden siliniyor

	cudaSetDevice(0);//İlk GPU aktif oluyor
	cudaFree(A1_GPU);//Dizi GPU1 belleğinden siliniyor
	cudaFree(B1_GPU);//Dizi GPU1 belleğinden siliniyor
	cudaFree(C1_GPU);//Dizi GPU1 belleğinden siliniyor

	cudaSetDevice(1);//İkinci GPU aktif oluyor
	cudaFree(A2_GPU);//Dizi GPU2 belleğinden siliniyor
	cudaFree(B2_GPU);//Dizi GPU2 belleğinden siliniyor
	cudaFree(C2_GPU);//Dizi GPU2 belleğinden siliniyor

	cudaSetDevice(2);//Üçüncü GPU aktif oluyor
	cudaFree(A3_GPU);//Dizi GPU3 belleğinden siliniyor
	cudaFree(B3_GPU);//Dizi GPU3 belleğinden siliniyor
	cudaFree(C3_GPU);//Dizi GPU3 belleğinden siliniyor
	
	cudaEventDestroy(start);//Event yok ediliyor
	cudaEventDestroy(stop);//Event yok ediliyor

	cudaError_t err = cudaGetLastError();//GPU'da oluşan son hatayı yakalıyor
	if ( err != cudaSuccess )
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
}

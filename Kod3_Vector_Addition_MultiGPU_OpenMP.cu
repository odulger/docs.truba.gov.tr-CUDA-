#include <stdio.h>
#include <omp.h>

__global__ void vector_addition(int *A,int *B,int *C,int size)//CUDA kernel
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	if(tid<size)
		C[tid] = A[tid] + B[tid];//Vector Addition gerçekleşiyor
}

int main()
{
	int size = 999999;//Dizi büyüklüğü (9'a tam bölünmeli)
	int ThreadPerBlock = 1024;//Blok büyüklüğü (Donanımın izin verdiği en yüksek değer)
	int BlockPerGrid = ((size/9)-1)/ThreadPerBlock+1;//Blok sayısı
	int *A_Host,*B_Host,*C_Host;

	cudaMallocHost((void**)&A_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	cudaMallocHost((void**)&B_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	cudaMallocHost((void**)&C_Host, sizeof(int)*size);//CPU belleğinde (Pinned) yer açılıyor
	
	for(int i=1;i<=size;i++)//Diziye başlangıç değerleri atanıyor
	{
		A_Host[i-1] = i;
		B_Host[i-1] = 0;
	}

	omp_set_num_threads(3);// GPU sayısı kadar thread oluşturulacak
	int tid;// Thread id
	
	int *A1_GPU[3],*B1_GPU[3],*C1_GPU[3];//Boyutu GPU sayısına eşit. İlgili GPU'daki 1. stream için kullanılacak
	int *A2_GPU[3],*B2_GPU[3],*C2_GPU[3];//Boyutu GPU sayısına eşit. İlgili GPU'daki 2. stream için kullanılacak
	int *A3_GPU[3],*B3_GPU[3],*C3_GPU[3];//Boyutu GPU sayısına eşit. İlgili GPU'daki 3. stream için kullanılacak
	
	cudaStream_t stream[3][3];//Satır büyüklüğü GPU sayısına, sutün büyüklüğü her GPU'daki stream sayısına eşit.
	
	for(int i=0;i<3;i++)//Her bir GPU için işlem yapacak
	{
		cudaSetDevice(i);//İlgili GPU aktif oluyor
		
		cudaStreamCreate(&stream[i][0]);//Aktif olan GPU'da 1.stream yaratılıyor
		cudaStreamCreate(&stream[i][1]);//Aktif olan GPU'da 2.stream yaratılıyor
		cudaStreamCreate(&stream[i][2]);//Aktif olan GPU'da 3.stream yaratılıyor
		
		cudaMalloc(&A1_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&B1_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&C1_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&A2_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&B2_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&C2_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&A3_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&B3_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
		cudaMalloc(&C3_GPU[i],sizeof(int)*size/9);//Aktif olan GPU'nun ana belleğinde yer açılıyor
	}
	
	dim3 DimBlock(ThreadPerBlock);//Bir bloktaki thread sayısı
	dim3 DimGrid(BlockPerGrid);//Bir griddeki blok sayısı
	
	double start = omp_get_wtime();//Başlangıç zamanı alınıyor

	#pragma omp parallel private(tid)//OpenMP paralel blok başlıyor
	{
		tid = omp_get_thread_num();//Thread id'si alınıyor
		cudaSetDevice(tid);//İlgili GPU aktif oluyor

		//Parçaların işlemleri CUDA Stream'ler kullanılarak overlap ediliyor
		cudaMemcpyAsync(A1_GPU[tid],A_Host+((tid*3)+0)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][0]);//CPU'dan GPU'ya veri aktarımı
		cudaMemcpyAsync(B1_GPU[tid],B_Host+((tid*3)+0)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][0]);//CPU'dan GPU'ya veri aktarımı
		vector_addition<<<DimGrid,DimBlock,0,stream[tid][0]>>>(A1_GPU[tid],B1_GPU[tid],C1_GPU[tid],size/9);//CUDA kernel çalıştırılıyor
		cudaMemcpyAsync(C_Host+((tid*3)+0)*(size/9),C1_GPU[tid],sizeof(int)*size/9,cudaMemcpyDeviceToHost,stream[tid][0]);//GPU'dan CPU'ya veri aktarımı

		cudaMemcpyAsync(A2_GPU[tid],A_Host+((tid*3)+1)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][1]);//CPU'dan GPU'ya veri aktarımı
		cudaMemcpyAsync(B2_GPU[tid],B_Host+((tid*3)+1)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][1]);//CPU'dan GPU'ya veri aktarımı
		vector_addition<<<DimGrid,DimBlock,0,stream[tid][1]>>>(A2_GPU[tid],B2_GPU[tid],C2_GPU[tid],size/9);//CUDA kernel çalıştırılıyor
		cudaMemcpyAsync(C_Host+((tid*3)+1)*(size/9),C2_GPU[tid],sizeof(int)*size/9,cudaMemcpyDeviceToHost,stream[tid][1]);//GPU'dan CPU'ya veri aktarımı

		cudaMemcpyAsync(A3_GPU[tid],A_Host+((tid*3)+2)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][2]);//CPU'dan GPU'ya veri aktarımı
		cudaMemcpyAsync(B3_GPU[tid],B_Host+((tid*3)+2)*(size/9),sizeof(int)*size/9,cudaMemcpyHostToDevice,stream[tid][2]);//CPU'dan GPU'ya veri aktarımı
		vector_addition<<<DimGrid,DimBlock,0,stream[tid][2]>>>(A3_GPU[tid],B3_GPU[tid],C3_GPU[tid],size/9);//CUDA kernel çalıştırılıyor
		cudaMemcpyAsync(C_Host+((tid*3)+2)*(size/9),C3_GPU[tid],sizeof(int)*size/9,cudaMemcpyDeviceToHost,stream[tid][2]);//GPU'dan CPU'ya veri aktarımı
		
		cudaStreamSynchronize(stream[tid][0]);//İlgili GPU'daki 1. stream'deki işlemler bitene kadar host bekliyor
		cudaStreamSynchronize(stream[tid][1]);//İlgili GPU'daki 2. stream'deki işlemler bitene kadar host bekliyor
		cudaStreamSynchronize(stream[tid][2]);//İlgili GPU'daki 3. stream'deki işlemler bitene kadar host bekliyor
	}
	
	double end = omp_get_wtime();//Bitiş zamanı alınıyor
	printf("Toplam Süre = %f saniye\n",end - start);//Geçen süre hesaplanıyor
	printf("C[size-1] = %d\n",C_Host[size-1]);
	
	cudaFreeHost(A_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(B_Host);//Dizi CPU belleğinden siliniyor
	cudaFreeHost(C_Host);//Dizi CPU belleğinden siliniyor
	
	for(int i=0;i<3;i++)//Her bir GPU için işlem yapacak
	{
		cudaSetDevice(i);//İlgili GPU aktif oluyor
		cudaFree(A1_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(B1_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(C1_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor

		cudaFree(A2_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(B2_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(C2_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor

		cudaFree(A3_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(B3_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
		cudaFree(C3_GPU[i]);//Aktif olan GPU'da Dizi GPU belleğinden siliniyor
	}
	
	
	for(int i=0;i<3;i++)//Her bir GPU için işlem yapacak
	{
		cudaSetDevice(i);//İlgili GPU aktif oluyor
		cudaStreamDestroy(stream[i][0]);//Aktif olan GPU'daki Stream 1 yok ediliyor
		cudaStreamDestroy(stream[i][1]);//Aktif olan GPU'daki Stream 2 yok ediliyor
		cudaStreamDestroy(stream[i][2]);//Aktif olan GPU'daki Stream 3 yok ediliyor
	}
	
	cudaError_t err = cudaGetLastError();//GPU'da oluşan son hatayı yakalıyor
	if ( err != cudaSuccess )
		printf("CUDA Error: %s\n",cudaGetErrorString(err));

}

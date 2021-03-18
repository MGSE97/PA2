// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>
#include <stdlib.h>

#include "imageKernels.cuh"

#define BLOCK_DIM 32
#define DATA_TYPE unsigned char

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

typedef struct image{
	DATA_TYPE* dData;
	unsigned int Width;
	unsigned int Height;
	unsigned int BPP;		//Bits Per Pixel = 8, 16, 2ColorToFloat_Channels, or 32 bit
	unsigned int Pitch;
} image_t;

KernelSetting ks;
KernelSetting ks2;

int* dResultsDataR = 0;
int* dResultsDataG = 0;
int* dResultsDataB = 0;
int* dResultsDataA = 0;

int* dResultsMaxR = 0;
int* dResultsMaxG = 0;
int* dResultsMaxB = 0;
int* dResultsMaxA = 0;

DATA_TYPE* dOutputDataR = 0;
DATA_TYPE* dOutputDataG = 0;
DATA_TYPE* dOutputDataB = 0;
DATA_TYPE* dOutputDataA = 0;

image loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
	tmp = FreeImage_ConvertTo8Bits(tmp);	// Large image fix

	image image;
	image.dData = 0;
	image.Width = FreeImage_GetWidth(tmp);
	image.Height = FreeImage_GetHeight(tmp);
	image.BPP = FreeImage_GetBPP(tmp);
	image.Pitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width
	
	checkCudaErrors(cudaMallocManaged((void**)&image.dData, image.Pitch * image.Height * image.BPP / 8));
	checkCudaErrors(cudaMemcpy(image.dData, FreeImage_GetBits(tmp), image.Pitch * image.Height * image.BPP / 8, cudaMemcpyHostToDevice));

	//checkHostMatrix<DATA_TYPE>(FreeImage_GetBits(tmp), image.Pitch, image.Height, image.Width, "%hhu ", "Result of Linear Pitch Text");
	//checkDeviceMatrix<DATA_TYPE>(image.dData, image.Pitch, image.Height, image.Width, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	//FreeImage_DeInitialise();

	return image;
}




void releaseMemory(image src)
{
	if (src.dData != 0)
		cudaFree(src.dData);

	if (dResultsDataR)
		cudaFree(dResultsDataR);
	if (dResultsDataG)
		cudaFree(dResultsDataG);
	if (dResultsDataB)
		cudaFree(dResultsDataB);
	if (dResultsDataA)
		cudaFree(dResultsDataA);

	if (dOutputDataR)
		cudaFree(dOutputDataR);
	if (dOutputDataG)
		cudaFree(dOutputDataG);
	if (dOutputDataB)
		cudaFree(dOutputDataB);
	if (dOutputDataA)
		cudaFree(dOutputDataA);

	FreeImage_DeInitialise();
}

__global__ void zeroKernel(
	int* src, const unsigned int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
		src[i] = 0;
}


__global__ void maxKernel(
	int* src, const unsigned int len, int* max)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
		atomicMax(max, src[i]);
}

__global__ void histogramKernel(
	DATA_TYPE* src, const unsigned int channel,
	const unsigned int width, const unsigned int height, const unsigned int pitch,
	int* result)
{	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col <= width && row <= height)
	{
		int c = src[col + row * pitch] & channel;
		atomicAdd(&result[c], 1);
	}
}

__global__ void histogram2DKernel(
	int* src, const unsigned int width, const unsigned int height, const int* limit, DATA_TYPE* result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width)
	{
		int v = src[x];
		double val = v / (double)*limit;
		v = height - height * val;
		for (int y = height - 1; y >= 0; y--)
			result[x  + y * width] = y >= v ? 255 : 0;
	}
}

void checkError(char* prefix)
{
	cudaDeviceSynchronize();
	auto ex = cudaGetLastError();
	if (ex != NULL)
		printf("Error at %s: %s\n", prefix, cudaGetErrorString(ex));
}

void saveChannel(std::string name, const int size, const int limit, DATA_TYPE* data)
{
	BYTE* result = (BYTE*)malloc(size * limit);
	checkCudaErrors(cudaMemcpy(result, data, size * limit, cudaMemcpyDeviceToHost));

	//checkHostMatrix(result, size, limit, size, "%d ");
	FIBITMAP* img = FreeImage_ConvertFromRawBits(result, limit, limit, size, 8, 0xFF, 0xFF, 0xFF);
	FreeImage_FlipVertical(img);
	ImageManager::GenericWriter(img, ("D:\\Documents\\Projekty\\Škola\\PA2\\cv9\\assets\\"+name+".png").c_str(), 0);

	FreeImage_Unload(img);
	SAFE_DELETE(result);
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	//ToDo: Edit to custom location
	image src = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv9\\assets\\lena.png");
	//image src = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv9\\assets\\RGB.png");


	printf("Loaded\n");

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((src.Width + BLOCK_DIM - 1) / BLOCK_DIM, (src.Height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	
	ks2.dimBlock = dim3(255, 1, 1);
	ks2.blockSize = 255;
	ks2.dimGrid = dim3(1, 1, 1);

	const int ch_size = 255 * sizeof(char);
	const int ch_limit = 100;
	const int cmp_size = 255 * sizeof(int);
	cudaMallocManaged((void**)&dResultsDataR, cmp_size);
	cudaMallocManaged((void**)&dResultsDataG, cmp_size);
	cudaMallocManaged((void**)&dResultsDataB, cmp_size);
	//cudaMallocManaged((void**)&dResultsDataA, cmp_size);

	cudaMallocManaged((void**)&dResultsMaxR, sizeof(int));
	cudaMallocManaged((void**)&dResultsMaxG, sizeof(int));
	cudaMallocManaged((void**)&dResultsMaxB, sizeof(int));
	//cudaMallocManaged((void**)&dResultsMaxA, sizeof(int));

	cudaMallocManaged((void**)&dOutputDataR, ch_size * ch_limit);
	cudaMallocManaged((void**)&dOutputDataG, ch_size * ch_limit);
	cudaMallocManaged((void**)&dOutputDataB, ch_size * ch_limit);
	//cudaMallocManaged((void**)&dOutputDataA, ch_size * ch_limit);
	
	checkError("Malloc");
	cudaEvent_t start, stop;
	float time;
	createTimer(&start, &stop, &time);

	startTimer(start);
	zeroKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataR, 255); checkError("Z-R");
	zeroKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataG, 255); checkError("Z-G");
	zeroKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataB, 255); checkError("Z-B");
	//zeroKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataA, 255); checkError("Z-A");

	histogramKernel<<<ks.dimGrid, ks.dimBlock>>>(src.dData, 224, src.Width, src.Height, src.Pitch, dResultsDataR); checkError("R");
	histogramKernel<<<ks.dimGrid, ks.dimBlock>>>(src.dData, 28, src.Width, src.Height, src.Pitch, dResultsDataG); checkError("G");
	histogramKernel<<<ks.dimGrid, ks.dimBlock>>>(src.dData, 3, src.Width, src.Height, src.Pitch, dResultsDataB); checkError("B");
	//histogramKernel<<<ks.dimGrid, ks.dimBlock>>>(src.dData, 3 << 6, src.Width, src.Height, src.Pitch, dResultsDataA); checkError("A");


	maxKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataR, 255, dResultsMaxR); checkError("M-R");
	maxKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataG, 255, dResultsMaxG); checkError("M-G");
	maxKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataB, 255, dResultsMaxB); checkError("M-B");
	//maxKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataA, 255, dResultsMaxA); checkError("M-A");

	histogram2DKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataR, 255, ch_limit, dResultsMaxR, dOutputDataR); checkError("2D-R");
	histogram2DKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataG, 255, ch_limit, dResultsMaxG, dOutputDataG); checkError("2D-G");
	histogram2DKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataB, 255, ch_limit, dResultsMaxB, dOutputDataB); checkError("2D-B");
	//histogram2DKernel<<<ks2.dimGrid, ks2.blockSize>>>(dResultsDataA, 255, ch_limit, dResultsMaxA, dOutputDataA); checkError("2D-A");

	stopTimer(start, stop, time);
	printf("Time: %f ms\n", time);

	saveChannel("R", ch_size, ch_limit, dOutputDataR);
	saveChannel("G", ch_size, ch_limit, dOutputDataG);
	saveChannel("B", ch_size, ch_limit, dOutputDataB);
	//saveChannel("A", ch_size, ch_limit, dOutputDataA);
	

	releaseMemory(src);
}

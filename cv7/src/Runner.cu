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

texture<DATA_TYPE, cudaTextureType2D, cudaReadModeElementType> refTex;		// declared texture reference must be at file-scope !!!
cudaChannelFormatDesc refTexChannelDesc;
size_t refTexPitch;

texture<DATA_TYPE, cudaTextureType2D, cudaReadModeElementType> queryTex;		// declared texture reference must be at file-scope !!!
cudaChannelFormatDesc queryTexChannelDesc;
size_t queryTexPitch;

typedef struct image{
	cudaArray* cuArray;
	DATA_TYPE* dData;
	unsigned int Width;
	unsigned int Height;
	unsigned int BPP;		//Bits Per Pixel = 8, 16, 2ColorToFloat_Channels, or 32 bit
	unsigned int Pitch;
} image_t;

KernelSetting ks;
KernelSetting ksr;
float3* dOutputData = 0;
float3* dResultsData = 0;

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

	checkCudaErrors(cudaMalloc((void**)&image.dData, image.Pitch * image.Height * image.BPP / 8));
	checkCudaErrors(cudaMemcpy(image.dData, FreeImage_GetBits(tmp), image.Pitch * image.Height * image.BPP / 8, cudaMemcpyHostToDevice));

	//checkHostMatrix<DATA_TYPE>(FreeImage_GetBits(tmp), image.Pitch, image.Height, image.Width, "%hhu ", "Result of Linear Pitch Text");
	//checkDeviceMatrix<DATA_TYPE>(image.dData, image.Pitch, image.Height, image.Width, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();

	return image;
}

void createTextures(image ref, image query)
{
	printf("Preparing textures ...\n");
	refTex.normalized = false;
	refTex.filterMode = cudaFilterModePoint;
	refTex.addressMode[0] = cudaAddressModeClamp;
	refTex.addressMode[1] = cudaAddressModeClamp;

	refTexChannelDesc = cudaCreateChannelDesc<DATA_TYPE>();

	checkCudaErrors(cudaMallocArray(&ref.cuArray, &refTexChannelDesc, ref.Pitch, ref.Height));
	checkCudaErrors(cudaMemcpyToArray(ref.cuArray, 0, 0, ref.dData, ref.Pitch * ref.Height * ref.BPP / 8 * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaBindTextureToArray(&refTex, ref.cuArray, &refTexChannelDesc));

	queryTex.normalized = false;
	queryTex.filterMode = cudaFilterModePoint;
	queryTex.addressMode[0] = cudaAddressModeClamp;
	queryTex.addressMode[1] = cudaAddressModeClamp;

	queryTexChannelDesc = cudaCreateChannelDesc<DATA_TYPE>();

	cudaMallocArray(&query.cuArray, &queryTexChannelDesc, query.Pitch, query.Height);
	cudaMemcpyToArray(query.cuArray, 0, 0, query.dData, query.Pitch * query.Height * ref.BPP / 8 * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice);

	checkCudaErrors(cudaBindTextureToArray(&queryTex, query.cuArray, &queryTexChannelDesc));
}


void releaseMemory(image ref, image querry)
{
	cudaUnbindTexture(refTex);
	cudaUnbindTexture(queryTex);
	
	if (ref.dData != 0)
		cudaFree(ref.dData);

	if (querry.dData != 0)
		cudaFree(querry.dData);
	
	if (dResultsData)
		cudaFree(dResultsData);
	if (dOutputData)
		cudaFree(dOutputData);
}

typedef struct position {
	int x;
	int y;
	float value;
} position_t;

// Using Texture units
__global__ void searchKernel(
	const unsigned int refWidth, const unsigned int refHeight,
	const unsigned int queryWidth, const unsigned int queryHeight,
	float3* tmp)
{
	__shared__ float3 sResults[BLOCK_DIM * BLOCK_DIM];
	unsigned int tidx = threadIdx.x;
	unsigned int tidy = threadIdx.y;
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int width = refWidth - queryWidth;
	unsigned int height = refHeight - queryHeight;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0;

	if (col <= width && row <= height)
	{
		for (int x = 0; x < queryWidth; x++)
			for (int y = 0; y < queryHeight; y++)
			{
				float u = col + x;
				float v = row + y;
				DATA_TYPE ref = tex2D(refTex, u, v);
				DATA_TYPE query = tex2D(queryTex, x, y);
				value += abs(ref - query);
			}
		sResults[tid].x = col;
		sResults[tid].y = row;
		sResults[tid].z = value;
	}
	else
		sResults[tid].z = INT32_MAX;

	__syncthreads();

	for (unsigned int y = blockDim.y >> 1; y > 0; y >>= 1)
	{
		if (tidy < y)
		{
			if (sResults[tid].z > sResults[tid + y * blockDim.x].z)
				sResults[tid] = sResults[tid + y * blockDim.x];
		}

		__syncthreads();
	}

	for (unsigned int x = blockDim.x >> 1; x > 0; x >>= 1)
	{
		if (tidx < x && tidy == 0)
		{
			if (sResults[tidx].z > sResults[tidx + x].z)
				sResults[tidx] = sResults[tidx + x];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		tmp[blockIdx.x + blockIdx.y * gridDim.x] = sResults[tid];
	}
}

// Using Image arrays
__global__ void searchKernel2(
	DATA_TYPE* refSrc, DATA_TYPE* querySrc,
	const unsigned int refWidth, const unsigned int refHeight, const unsigned int refPitch,
	const unsigned int queryWidth, const unsigned int queryHeight, const unsigned int queryPitch,
	float3* tmp)
{
	__shared__ float3 sResults[BLOCK_DIM * BLOCK_DIM];
	unsigned int tidx = threadIdx.x;
	unsigned int tidy = threadIdx.y;
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int width = refWidth - queryWidth;
	unsigned int height = refHeight - queryHeight;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0;

	if (col <= width && row <= height)
	{
		for (unsigned int x = 0; x < queryWidth; x++)
			for (unsigned int y = 0; y < queryHeight; y++)
			{
				unsigned int u = col + x;
				unsigned int v = row + y;
				DATA_TYPE ref = refSrc[v * refPitch + u];
				DATA_TYPE query = querySrc[y * queryPitch + x];
				value += abs(ref - query);
			}
		sResults[tid].x = col;
		sResults[tid].y = row;
		sResults[tid].z = value;
	}
	else
		sResults[tid].z = INT32_MAX;

	__syncthreads();

	for (unsigned int y = blockDim.y >> 1; y > 0; y >>= 1)
	{
		if (tidy < y)
		{
			if (sResults[tid].z > sResults[tid + y * blockDim.x].z)
				sResults[tid] = sResults[tid + y * blockDim.x];
		}

		__syncthreads();
	}

	for (unsigned int x = blockDim.x >> 1; x > 0; x >>= 1)
	{
		if (tidx < x && tidy == 0)
		{
			if (sResults[tidx].z > sResults[tidx + x].z)
				sResults[tidx] = sResults[tidx + x];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		tmp[blockIdx.x + blockIdx.y * gridDim.x] = sResults[tid];
	}
}

// Final reduction between blocks
__global__ void reduceKernel(
	float3* src,
	unsigned int len,
	float3* dst)
{
	__shared__ float3 sResults[BLOCK_DIM * BLOCK_DIM];
	unsigned int tidx = threadIdx.x;
	unsigned int tidy = threadIdx.y;
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	if (tid == 0)
	{
		dst->z = (float)INT32_MAX;
	}

	unsigned int offset = 0;
	while (offset < len)
	{
		if (tid < len)
			sResults[tid] = src[tid + offset];
		else
			sResults[tid].z = (float)INT32_MAX;

		__syncthreads();

		for (unsigned int y = blockDim.y >> 1; y > 0; y >>= 1)
		{
			if (tidy < y)
			{
				if (sResults[tid].z > sResults[tid + y * blockDim.x].z)
					sResults[tid] = sResults[tid + y * blockDim.x];
			}

			__syncthreads();
		}

		for (unsigned int x = blockDim.x >> 1; x > 0; x >>= 1)
		{
			if (tidx < x && tidy == 0)
			{
				if (sResults[tidx].z > sResults[tidx + x].z)
					sResults[tidx] = sResults[tidx + x];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			if(sResults[tidx].z < dst->z)
				*dst = sResults[tidx];
		}

		offset += blockDim.x * blockDim.y;
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	//ToDo: Edit to custom location
	image reference = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv7\\assets\\reference.tif");
	image querry = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv7\\assets\\query.tif");
	/*image reference = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv7\\assets\\reference_s.tiff");
	image querry = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv7\\assets\\query_s.tiff");
	image querry = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv7\\assets\\query_s2.tiff");*/

	printf("Loaded\n");

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((reference.Width - querry.Width + BLOCK_DIM - 1) / BLOCK_DIM, (reference.Height - querry.Height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	ks.sharedMemSize = ks.blockSize;

	ksr.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ksr.blockSize = BLOCK_DIM * BLOCK_DIM;
	ksr.dimGrid = dim3(1, 1, 1);//dim3((ks.dimGrid.x + BLOCK_DIM - 1) / BLOCK_DIM, (ks.dimGrid.y + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	ksr.sharedMemSize = ksr.blockSize;

	cudaMalloc((void**)&dResultsData, ks.dimGrid.x * ks.dimGrid.y * sizeof(float3));
	cudaMalloc((void**)&dOutputData, sizeof(float3));

	cudaEvent_t start, stop;
	float time;
	createTimer(&start, &stop, &time);

	printf("\nSearching using arrays ...\n");
	startTimer(start);
	searchKernel2<<<ks.dimGrid, ks.dimBlock, ks.sharedMemSize>>>(reference.dData, querry.dData, reference.Width, reference.Height, reference.Pitch, querry.Width, querry.Height, querry.Pitch, dResultsData);
	reduceKernel<<<ksr.dimGrid, ksr.dimBlock, ksr.sharedMemSize>>>(dResultsData, ks.dimGrid.x * ks.dimGrid.y, dOutputData);
	auto ex = cudaGetLastError();
	if (ex != NULL)
		printf("Error: %s\n", cudaGetErrorString(ex));
	stopTimer(start, stop, time);
	printf("Search completed\n");
	printf("Time: %f ms\n", time);

	float3 result;
	checkCudaErrors(cudaMemcpy(&result, dOutputData, sizeof(float3), cudaMemcpyDeviceToHost));

	printf("Pattern location: %5.0f, %5.0f\n", result.x, reference.Height - result.y - querry.Height);
	printf("Pattern difference: %9.0f\n\n", result.z);
	

	createTextures(reference, querry);
	printf("Searching using textures ...\n");
	startTimer(start);
	searchKernel<<<ks.dimGrid, ks.dimBlock, ks.sharedMemSize>>>(reference.Width, reference.Height, querry.Width, querry.Height, dResultsData);
	reduceKernel<<<ksr.dimGrid, ksr.dimBlock, ksr.sharedMemSize>>>(dResultsData, ks.dimGrid.x * ks.dimGrid.y, dOutputData);
	ex = cudaGetLastError();
	if(ex != NULL)
		printf("Error: %s\n", cudaGetErrorString(ex));
	stopTimer(start, stop, time);
	printf("Search completed\n");
	printf("Time: %f ms\n", time);

	checkCudaErrors(cudaMemcpy(&result, dOutputData, sizeof(float3), cudaMemcpyDeviceToHost));

	printf("Pattern location: %5.0f, %5.0f\n", result.x, reference.Height - result.y - querry.Height);
	printf("Pattern difference: %9.0f\n", result.z);

	releaseMemory(reference, querry);
}

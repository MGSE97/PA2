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

texture<DATA_TYPE, cudaTextureType2D, cudaReadModeElementType> tex;		// declared texture reference must be at file-scope !!!
cudaChannelFormatDesc texChannelDesc;
size_t texPitch;

typedef struct image{
	cudaArray* cuArray;
	DATA_TYPE* dData;
	unsigned int Width;
	unsigned int Height;
	unsigned int BPP;		//Bits Per Pixel = 8, 16, 2ColorToFloat_Channels, or 32 bit
	unsigned int Pitch;
} image_t;

KernelSetting ks;
DATA_TYPE* dResultsData = 0;

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

	return image;
}

void createTextures(image ref)
{
	printf("Preparing textures ...\n");
	tex.normalized = false;
	tex.filterMode = cudaFilterModePoint;
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;

	texChannelDesc = cudaCreateChannelDesc<DATA_TYPE>();

	checkCudaErrors(cudaMallocArray(&ref.cuArray, &texChannelDesc, ref.Pitch, ref.Height));
	checkCudaErrors(cudaMemcpyToArray(ref.cuArray, 0, 0, ref.dData, ref.Pitch * ref.Height * ref.BPP / 8 * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaBindTextureToArray(&tex, ref.cuArray, &texChannelDesc));
}


void releaseMemory(image ref)
{
	cudaUnbindTexture(tex);
	
	if (ref.dData != 0)
		cudaFree(ref.dData);
	
	if (dResultsData)
		cudaFree(dResultsData);

	FreeImage_DeInitialise();
}

// Using Texture units
__global__ void normalKernel(
	const unsigned int Width, const unsigned int Height,
	const float Power,
	DATA_TYPE* result)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = (row * Width + col) * 3;

	float sobelX[] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};
	float sobelY[] = {
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1
	};

	if (col < Width && row < Height)
	{
		float valueX = 0, valueY = 0;
		for (int y = 0; y < 3; y++)
			for (int x = 0; x < 3; x++)
			{
				int o = x + y * 3;
				float u = col + x - 1;
				float v = row - y + 1;
				float img = tex2D(tex, u, v);
				valueX += img * sobelX[o];
				valueY += img * sobelY[o];
			}

		float n = sqrt(valueX * valueX + valueY * valueY + Power * Power);
		result[tid+2] = valueX / n * 255.f;
		result[tid+1] = valueY / n * 255.f;
		result[tid] = Power / n * 255.f;
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	//ToDo: Edit to custom location
	image image = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv8\\assets\\terrain3Kx3K.tif");
	//image image = loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv8\\assets\\brick_bump.png");


	printf("Loaded\n");

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((image.Width + BLOCK_DIM - 1) / BLOCK_DIM, (image.Height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	ks.sharedMemSize = ks.blockSize;

	cudaMallocManaged((void**)&dResultsData, image.Width * image.Height * sizeof(DATA_TYPE) * 3);

	cudaEvent_t start, stop;
	float time;
	createTimer(&start, &stop, &time);
	BYTE* result = (BYTE*)malloc(image.Width * image.Height * sizeof(DATA_TYPE) * 3);

	createTextures(image);
	printf("Converting using textures ...\n");
	startTimer(start);
	normalKernel<<<ks.dimGrid, ks.dimBlock>>>(image.Width, image.Height, 0.5f, dResultsData);
	stopTimer(start, stop, time);
	auto ex = cudaGetLastError();
	if (ex != NULL)
		printf("Error: %s\n", cudaGetErrorString(ex));
	printf("Conversion completed\n");
	printf("Time: %f ms\n", time);

	//checkDeviceMatrix(dResultsData, image.Width, 10, 30, "%3d ");

	checkCudaErrors(cudaMemcpy(result, dResultsData, image.Width * image.Height * sizeof(DATA_TYPE) * 3, cudaMemcpyDeviceToHost));

	printf("Saved\n");
	auto img = FreeImage_ConvertFromRawBits(result, image.Width, image.Height, image.Width * sizeof(DATA_TYPE) * 3, 24, 0xFF, 0xFF, 0xFF);
	ImageManager::GenericWriter(img, "D:\\Documents\\Projekty\\Škola\\PA2\\cv8\\assets\\terrain3Kx3K_normals.tif", 0);
	//ImageManager::GenericWriter(img, "D:\\Documents\\Projekty\\Škola\\PA2\\cv8\\assets\\brick_normals.png", 0);
	FreeImage_Unload(img);

	releaseMemory(image);
}

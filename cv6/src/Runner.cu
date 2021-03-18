// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>
#include <stdlib.h>

#include "imageKernels.cuh"

#define BLOCK_DIM 8
#define ColorToFloat_Channels 4

template __global__ void colorToFloat<8>(const unsigned char* __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst);
template __global__ void colorToFloat<16>(const unsigned char* __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst);
template __global__ void colorToFloat<24>(const unsigned char* __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst);
template __global__ void colorToFloat<32>(const unsigned char* __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst);

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;		// declared texture reference must be at file-scope !!!

cudaChannelFormatDesc texChannelDesc;

unsigned char *dImageData = 0;
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 2ColorToFloat_Channels, or 32 bit
unsigned int imagePitch;

size_t texPitch;
float *dLinearPitchTextureData = 0;
cudaArray *dArrayTextureData = 0;

KernelSetting ks;

float *dOutputData = 0;

void loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width

	cudaMalloc((void**)&dImageData, imagePitch * imageHeight * imageBPP/8);
	cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight * imageBPP/8, cudaMemcpyHostToDevice);

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

void createTextureFromLinearPitchMemory()
{
	// TODO: Allocate dLinearPitchTextureData variable memory
	//																				width, height * rgba
	checkCudaErrors(cudaMallocPitch((void**)&dLinearPitchTextureData, &texPitch, imageWidth, imageHeight * ColorToFloat_Channels));
	printf("\nImagePitch: %d\nTexPitch: %d\nImageBPP: %d\n", imagePitch, texPitch, imageBPP);
	switch(imageBPP)
	{
		//TODO: Here call your kernel to convert image into linearPitch memory
		case 8:
			colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearPitchTextureData);
			break;
		case 16:
			colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearPitchTextureData);
			break;
		case 24:
			colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearPitchTextureData);
			break;
		case 32:
			colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearPitchTextureData);
			break;
	}

	checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight * ColorToFloat_Channels, imageWidth, "%6.1f ", "Result of Linear Pitch Text");

	//TODO: Define texture (texRef) parameters
	texRef.normalized = false;
	texRef.filterMode = cudaFilterModePoint;
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;

	//TODO: Define texture channel descriptor (texChannelDesc)
	texChannelDesc = cudaCreateChannelDesc<float>();
	
	//TODO: Bind texture
	cudaBindTexture2D(0, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight * ColorToFloat_Channels, texPitch);

}

void createTextureFrom2DArray()
{
	//TODO: Define texture (texRef) parameters
	texRef.normalized = false;
	texRef.filterMode = cudaFilterModePoint;
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;

	//TODO: Define texture channel descriptor (texChannelDesc)
	texChannelDesc = cudaCreateChannelDesc<float>();

	//Converts custom image data to float and stores result in the float_linear_data
	float *dLinearTextureData = 0;
	//Disabled pith = set to data width
	texPitch = imageWidth * sizeof(float);
	//										width * height * rgba * data_type
	cudaMalloc((void**)&dLinearTextureData, imageWidth * imageHeight * ColorToFloat_Channels * sizeof(float));
	switch(imageBPP)
	{
		//TODO: Here call your kernel to convert image into linear memory (no pitch!!!)
		case 8:
			colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearTextureData);
			break;
		case 16:
			colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearTextureData);
			break;
		case 24:
			colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearTextureData);
			break;
		case 32:
			colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch, dLinearTextureData);
			break;
	}

	checkDeviceMatrix<float>(dLinearTextureData, texPitch, imageHeight * ColorToFloat_Channels, imageWidth, "%6.1f ", "Result of Linear Text");

	cudaMallocArray(&dArrayTextureData, &texChannelDesc, imageWidth, imageHeight * ColorToFloat_Channels);
	
	//TODO: copy data into cuda array (dArrayTextureData)
	cudaMemcpyToArray(dArrayTextureData, 0, 0, dLinearTextureData, imageWidth*imageHeight*ColorToFloat_Channels*sizeof(float), cudaMemcpyDeviceToDevice);
	
	//TODO: Bind texture
	cudaBindTextureToArray(&texRef, dArrayTextureData, &texChannelDesc);

	cudaFree(dLinearTextureData);
}


void releaseMemory()
{
	cudaUnbindTexture(texRef);
	if (dImageData!=0)
		cudaFree(dImageData);
	if (dLinearPitchTextureData!=0)
		cudaFree(dLinearPitchTextureData);
	if (dArrayTextureData)
		cudaFreeArray(dArrayTextureData);
	if (dOutputData)
		cudaFree(dOutputData);
}


__global__ void texKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < texWidth && row < texHeight)
	{
		float u = col;
		float v = row * ColorToFloat_Channels;

		dst[row * texWidth + col] = tex2D(texRef, u, v);
	}
}


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	//ToDo: Edit to custom location
	loadSourceImage("D:\\Documents\\Projekty\\Škola\\PA2\\cv6\\assets\\terrain10x10.tif");

	cudaMalloc((void**)&dOutputData, imageWidth * imageHeight * sizeof(float));

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((imageWidth + BLOCK_DIM-1)/BLOCK_DIM, (imageHeight + BLOCK_DIM-1)/BLOCK_DIM, 1);

	//Test 1 - texture stored in linear pitch memory
	createTextureFromLinearPitchMemory();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

	//Test 2 - texture stored in 2D array
	createTextureFrom2DArray();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData,  imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

	releaseMemory();
}

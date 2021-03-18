// includes, cuda
#include <cstdint>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cudaDefs.h>
#include "arrayUtils.cuh"

#define TPB 256

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

KernelSetting ks;

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	const size_t srcWidth = 25;
	const size_t srcHeight = 4;
	const size_t srcPitchInBytes = 512;
	const size_t srcSize = srcHeight * srcPitchInBytes;
	uint8_t* src = nullptr;
	cudaMallocManaged(&src, srcSize);

	const size_t dstWidth = 50;
	const size_t dstHeight = 2;
	const size_t dstPitchInBytes = 512; //dstWidth * sizeof(float);
	const size_t dstSize = dstHeight * dstPitchInBytes;
	float* dst= nullptr;
	cudaMallocManaged(&dst, dstSize);

	for (size_t i = 0; i < srcWidth * srcHeight; i++)
	{
		size_t y = i / srcWidth;
		size_t x = i - y * srcWidth;
		uint8_t* srcPtr = (uint8_t*)((char*)src + y * srcPitchInBytes) + x;
		*srcPtr = static_cast<uint8_t>(i);
	}

	checkHostMatrix(src, srcPitchInBytes, srcHeight, srcWidth, "%d ", "SRC");
	checkDeviceMatrix(src, srcPitchInBytes, srcHeight, srcWidth, "%d ", "SRC");

	ks.dimBlock = dim3(TPB, 1, 1);
	ks.blockSize = TPB;
	ks.dimGrid = dim3(getNumberOfParts(srcWidth * srcHeight, TPB),1,1);

	arrayReshape<uint8_t, float> <<<ks.dimGrid, ks.dimBlock>>> (src, srcWidth, srcHeight, srcPitchInBytes, dstWidth, dstHeight, dstPitchInBytes, dst);
	cudaDeviceSynchronize();

	checkHostMatrix(dst, dstPitchInBytes, dstHeight, dstWidth, "%f ", "DST");
	checkDeviceMatrix(dst, dstPitchInBytes, dstHeight, dstWidth, "%f ", "DST");

	cudaFree(src);
	cudaFree(dst);
}

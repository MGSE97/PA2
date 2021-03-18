#include <cudaDefs.h>
#include <time.h>
#include <math.h>

#define RAND false

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

const unsigned int N = 1 << 20;
const unsigned int MEMSIZE = N * sizeof(unsigned int);
const unsigned int NO_LOOPS = 1000;
const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int GRID_SIZE = (N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;

void fillData(unsigned int *data, const unsigned int length)
{
#if RAND
	srand(time(0));
	for (unsigned int i = 0; i < length; i++)
		data[i] = rand();
#else
	for (unsigned int i = 0; i < length; i++)
		data[i]= 1;
#endif
}

void printData(const unsigned int *data, const unsigned int length)
{
	if (data ==0) return;
	for (unsigned int i=0; i<length; i++)
	{
		printf("%u ", data[i]);
	}
}


__global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < length)
		c[tid] = a[tid] + b[tid];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc( (void**)&da, MEMSIZE );
	cudaMalloc( (void**)&db, MEMSIZE );
	cudaMalloc( (void**)&dc, MEMSIZE );

	// create stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	unsigned int dataOffset = 0;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for(int i=0; i < NO_LOOPS; i++)
	{
		// Do copy kernel copy
		cudaMemcpyAsync(da, a, MEMSIZE, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(db, b, MEMSIZE, cudaMemcpyHostToDevice, stream);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream>>>(da, db, N, dc);
		cudaMemcpyAsync(c, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);
		dataOffset += N;
	}

	// Wait for it and destroy
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("\nTest time: %f ms\n", elapsedTime);

	printData(c, 100);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	unsigned int* a, * b, * c;
	unsigned int* da0, * db0, * dc0;
	unsigned int* da1, * db1, * dc1;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da0, MEMSIZE);
	cudaMalloc((void**)&db0, MEMSIZE);
	cudaMalloc((void**)&dc0, MEMSIZE);

	cudaMalloc((void**)&da1, MEMSIZE);
	cudaMalloc((void**)&db1, MEMSIZE);
	cudaMalloc((void**)&dc1, MEMSIZE);

	// create stream
	cudaStream_t stream0;
	cudaStreamCreate(&stream0);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	unsigned int dataOffset = 0;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		//stream 0
		cudaMemcpyAsync(da0, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db0, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream0>>>(da0, db0, N, dc0);
		cudaMemcpyAsync(&c[dataOffset], dc0, MEMSIZE, cudaMemcpyDeviceToHost, stream0);
		dataOffset += N;

		//stream 1
		cudaMemcpyAsync(da1, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db1, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream1>>>(da1, db1, N, dc1);
		cudaMemcpyAsync(&c[dataOffset], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1);
		dataOffset += N;
	}

	// Wait for it and destroy
	cudaStreamSynchronize(stream0);
	cudaStreamDestroy(stream0);

	cudaStreamSynchronize(stream1);
	cudaStreamDestroy(stream1);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("\nTest time: %f ms\n", elapsedTime);

	printData(c, 100);

	cudaFree(da0);
	cudaFree(db0);
	cudaFree(dc0);

	cudaFree(da1);
	cudaFree(db1);
	cudaFree(dc1);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	unsigned int* a, * b, * c;
	unsigned int* da0, * db0, * dc0;
	unsigned int* da1, * db1, * dc1;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da0, MEMSIZE);
	cudaMalloc((void**)&db0, MEMSIZE);
	cudaMalloc((void**)&dc0, MEMSIZE);

	cudaMalloc((void**)&da1, MEMSIZE);
	cudaMalloc((void**)&db1, MEMSIZE);
	cudaMalloc((void**)&dc1, MEMSIZE);

	// create stream
	cudaStream_t stream0;
	cudaStreamCreate(&stream0);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	fillData(b, NO_LOOPS * N);

	unsigned int dataOffset0 = 0;
	unsigned int dataOffset1 = N;
	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		cudaMemcpyAsync(da0, &a[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(da1, &a[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db0, &b[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db1, &b[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream0>>>(da0, db0, N, dc0);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream1>>>(da1, db1, N, dc1);
		cudaMemcpyAsync(&c[dataOffset0], dc0, MEMSIZE, cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(&c[dataOffset1], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1);
		dataOffset0 += 2 * N;
		dataOffset1 += 2 * N;
	}

	// Wait for it and destroy
	cudaStreamSynchronize(stream0);
	cudaStreamDestroy(stream0);

	cudaStreamSynchronize(stream1);
	cudaStreamDestroy(stream1);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("\nTest time: %f ms\n", elapsedTime);

	printData(c, 100);

	cudaFree(da0);
	cudaFree(db0);
	cudaFree(dc0);

	cudaFree(da1);
	cudaFree(db1);
	cudaFree(dc1);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	test1();
	test2();
	test3();

	return 0;
}

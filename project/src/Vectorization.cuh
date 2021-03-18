#pragma once
#include <cuda_runtime.h>
#include <cudaDefs.h>
#include "Settings.h"

//void setSearchLength(unsigned int search_length, cudaStream_t& stream);
//void setLineSize(unsigned int line_size, cudaStream_t& stream);

__global__ void clearArray(
	unsigned char* arr,
	const unsigned int lenght);

__global__ void splitLine(
	unsigned char* __restrict__ lines,
	const unsigned int  lines_lenght,
	const unsigned int  line_size,
	unsigned char* strings,
	unsigned int* lengths);

__global__ void findPattern(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int search_size,
	unsigned char* __restrict__ patterns,
	const unsigned int patterns_lenght,
	const unsigned int results_offset,
	RESULT_TYPE* results);

__global__ void countChars(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int  search_size,
	const unsigned int results_offset,
	RESULT_TYPE* results);

__global__ void stringToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int  search_size,
	const unsigned int results_offset,
	RESULT_TYPE* results);

__global__ void dateToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int results_offset,
	RESULT_TYPE* results);

__global__ void copyTo(
	unsigned int* __restrict__ input,
	const unsigned int intput_padding,
	const unsigned int intput_offset,
	const unsigned int intput_length,
	const unsigned int change,
	const unsigned int results_padding,
	const unsigned int results_offset,
	RESULT_TYPE* results);

__global__ void clasify(
	RESULT_TYPE* results,
	const unsigned int lines,
	const unsigned int classificator_offset,
	const unsigned int resultS_offset);

__global__ void sanitize(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int search_size,
	const unsigned int results_offset,
	unsigned char* results);
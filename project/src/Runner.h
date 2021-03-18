#pragma once
#include <cuda_runtime.h>
#include <cudaDefs.h>

#include "Settings.h"
#include "File.h"
#include "CPU.h"
#include "Settings.h"

#include "Vectorization.cuh"

typedef struct StreamData {
	cudaStream_t stream;
	unsigned char* data;

	unsigned char* d_lines;
	unsigned char* d_strings;
	unsigned int* d_lengths;

	RESULT_TYPE* h_data;
	RESULT_TYPE* d_results;

	unsigned char* h_texts;
	unsigned char* d_text_results;

	unsigned int h_lines;
};

void RunStream(unsigned int& lines, unsigned char* file_data, unsigned int& full_dim, std::ifstream* requests, std::ofstream* results, std::ofstream* clasifications, StreamData* data, unsigned int& sum, unsigned int* items,
	unsigned char* d_safe, unsigned int& safe_length, unsigned char* d_neutral, unsigned int& neutral_length, unsigned char* d_invalid, unsigned int& invalid_length,
	unsigned char* d_trusted, unsigned int& trusted_length, unsigned char* d_search, unsigned int& search_length);
void CreateStream(StreamData* data);
void DisposeStream(StreamData* data);

void computeVectors();

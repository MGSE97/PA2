// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include "Runner.h"

#define BLOCK_DIM 512
#define BLOCK_DIM_2D 32
//#define LINES 4096
#define LINES 8196
#define ITEM_SIZE 4096
#define LINE_PARTS 7
#define LINE_SIZE LINE_PARTS * ITEM_SIZE
#define EMPTY_CHAR 205
#define RESULT_VALUES 18
#define RESULT_TYPE unsigned int

cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

vector<string> split(const string& str, const string& delim)
{
	vector<string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == string::npos) pos = str.length();
		string token = str.substr(prev, pos - prev);
		if (!token.empty()) tokens.push_back(token);
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());
	return tokens;
}

ifstream* openFile(const char* file)
{
	ifstream* stream = new ifstream();
	stream->open(file);
	return stream;
}

ofstream* writeFile(const char* file)
{
	ofstream* stream = new ofstream();
	// stream->open(file, ofstream::app); // Append
	stream->open(file, ofstream::trunc); // Rewrite
	return stream;
}

void closeFile(ifstream* stream)
{
	if (stream->is_open())
		stream->close();
}

void closeFile(ofstream* stream)
{
	if (stream->is_open())
		stream->close();
}

unsigned int skipFile(ifstream* stream, const unsigned int lines)
{
	if (!stream->is_open())
		throw exception("Stream is not open!");

	string line;
	bool success = true;
	unsigned int i = 0;
	for (; i < lines && success; i++)
	{
		success = !getline(*stream, line).eof();
	}
	return i;
}

unsigned int readFile(ifstream* stream, unsigned char* data, const unsigned int lines)
{
	if (!stream->is_open())
		throw exception("Stream is not open!");

	string line;
	bool success = true;
	unsigned int len = 0;
	for (unsigned int i = 0; i < lines && success; i++)
	{
		success = !getline(*stream, line).eof();
		if (success)
		{
			memcpy(&data[i * LINE_SIZE], line.c_str(), line.size() * sizeof(char));
			len++;
		}
	}
	return len;
}

void appendFile(ofstream* stream, RESULT_TYPE* data, const unsigned int lines)
{
	if (lines == 0)
		return;

	if (!stream->is_open())
		throw exception("Stream is not open!");

	string write;
	for (unsigned int i = 0; i < lines; i++)
	{
		unsigned int o = i * RESULT_VALUES;
		for (unsigned int j = 0; j < RESULT_VALUES - 1; j++)
		{
			write += to_string(data[o + j]) + ",";
		}
		write += to_string(data[o + RESULT_VALUES - 1]) + "\n";
	}
	stream->write(write.c_str(), write.size());
}


// lines
__global__ void clearArray(
	unsigned char* arr,
	const unsigned long lenght)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = gridDim.x * blockDim.x;

	while (offset < lenght)
	{
		arr[offset] = 0;
		offset += skip;
	}
}

// lines
__global__ void splitLine(
	 unsigned char* __restrict__ lines,
	const unsigned long  lines_lenght,
	unsigned char* strings,
	unsigned int* lengths)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < lines_lenght)
	{
		unsigned int ol = offset * LINE_SIZE;
		unsigned int op = offset * LINE_PARTS;

		unsigned char split = ',', escape = '\"';

		unsigned int part = 0;
		unsigned int len = 0;
		unsigned int pos = 0;

		unsigned char c_old = 0;
		bool escaped = false;
		bool do_split = false;

		for (unsigned int i = 0; pos < LINE_SIZE; i++)
		{
			char c = lines[ol + i];

			strings[ol + pos] = c;

			escaped = (c_old == escape && c == split) ? false : escaped;
			escaped = (c_old == split && c == escape) ? true : escaped;
			do_split = c == split && !escaped;
			
			part += do_split ? 1 : 0;
			len = do_split ? 0 : (len + 1);

			lengths[op + part] = len;

			pos = part * ITEM_SIZE + len;

			c_old = c;
		}
	}
}

__global__ void findPattern(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned long strings_lenght,
	unsigned char* __restrict__ patterns,
	const unsigned long patterns_lenght,
	const unsigned int results_padding,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset_string = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset_string < strings_lenght)
	{
		unsigned int result = 0;
		for (unsigned int pattern = 0; pattern < patterns_lenght; pattern++)
		{
			// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
			//			------------ LINE ----------, -ITEM-, -ITEM-
			unsigned int os = offset_string * LINE_SIZE + part * ITEM_SIZE;
			unsigned int op = pattern * ITEM_SIZE;

			unsigned int p = 0;

			bool found = false;
			for (unsigned int i = 0; i < ITEM_SIZE; i++)
			{
				unsigned char a = strings[os + i],
							  b = patterns[op + p];

				found = (b == EMPTY_CHAR) ? (p > 0) : found;
				p = (a == b) ? (p + 1) : 0;
			}

			result += found;
		}

		results[offset_string * results_padding + results_offset] = result > 0;
	}
}

__global__ void countChars(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned long strings_lenght,
	const unsigned int results_padding,
	const unsigned int results_offset_lowerCase,
	const unsigned int results_offset_upperCase,
	const unsigned int results_offset_numbers,
	const unsigned int results_offset_specials,
	const unsigned int results_offset_whites,
	RESULT_TYPE* results)
{
	unsigned int offset_string = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset_string < strings_lenght)
	{

		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset_string * LINE_SIZE + part * ITEM_SIZE;

		char whites_c[] = { ' ', '\f', '\n', '\r', '\t', '\v' };
		unsigned int whites_len = 6;

		RESULT_TYPE lowerCase = 0;
		RESULT_TYPE upperCase = 0;
		RESULT_TYPE numbers = 0;
		RESULT_TYPE specials = 0;
		RESULT_TYPE whites = 0;

		for (unsigned int i = 0; i < ITEM_SIZE; i++)
		{
			unsigned char c = strings[os + i];

			lowerCase += (c >= 'a' && c <= 'z') ? 1 : 0;
			upperCase += (c >= 'A' && c <= 'Z') ? 1 : 0;
			numbers += (c >= '0' && c <= '9') ? 1 : 0;

			bool is_white = false;
			for (unsigned int w = 0; w < whites_len; w++) {
				whites += c == whites_c[w] ? 1 : 0;
				is_white = c == whites_c[w] ? true : is_white;
			}

			bool is_not_special = is_white;
			is_not_special = (c >= 'a' && c <= 'z') ? true : is_not_special;
			is_not_special = (c >= 'A' && c <= 'Z') ? true : is_not_special;
			is_not_special = (c >= '0' && c <= '9') ? true : is_not_special;
			is_not_special = c == EMPTY_CHAR ? true : is_not_special;
			is_not_special = c == 0 ? true : is_not_special;
			
			specials += is_not_special ? 0 : 1;
		}

		results[offset_string * results_padding + results_offset_lowerCase] = lowerCase;
		results[offset_string * results_padding + results_offset_upperCase] = upperCase;
		results[offset_string * results_padding + results_offset_numbers]   = numbers;
		results[offset_string * results_padding + results_offset_specials]  = specials;
		results[offset_string * results_padding + results_offset_whites]    = whites;
	}
}

// lines
__global__ void stringToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned long strings_lenght,
	const unsigned int results_padding,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset * LINE_SIZE + part * ITEM_SIZE;

		unsigned char zero = '0';
		unsigned char nine = '9';

		RESULT_TYPE val = 0;

		// get int
		for (unsigned int i = 0; i < ITEM_SIZE; i++)
		{
			unsigned char c = strings[os + i];
			val = (c >= zero && c <= nine) ? (val * 10 + c - zero) : val;
		}

		results[offset * results_padding + results_offset] = val;
	}
}

// lines
__global__ void dateToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned long strings_lenght,
	const unsigned int results_padding,
	const unsigned int results_offset_y,
	const unsigned int results_offset_m,
	const unsigned int results_offset_d,
	const unsigned int results_offset_H,
	const unsigned int results_offset_M,
	const unsigned int results_offset_S,
	RESULT_TYPE* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset * LINE_SIZE + part * ITEM_SIZE;

		unsigned char zero = '0';
		unsigned char nine = '9';

		RESULT_TYPE values[8];
		int type = -1;

		// get int
		for (unsigned int i = 0; i < 20; i++)
		{
			unsigned char c = strings[os + i];
			bool number = c >= zero && c <= nine;
			type += number ? 0 : 1;
			values[type] = number ? (values[type] * 10 + c - zero) : 0;
		}

		results[offset * results_padding + results_offset_y] = values[0];
		results[offset * results_padding + results_offset_m] = values[1];
		results[offset * results_padding + results_offset_d] = values[2];
		results[offset * results_padding + results_offset_H] = values[3];
		results[offset * results_padding + results_offset_M] = values[4];
		results[offset * results_padding + results_offset_S] = values[5];
	}
}

// lines
__global__ void copyTo(
	unsigned int* __restrict__ input,
	const unsigned int intput_padding,
	const unsigned int intput_offset,
	const unsigned int intput_length,
	const unsigned int change,
	const unsigned int results_padding,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = gridDim.x * blockDim.x;

	while (offset < intput_length)
	{
		unsigned int oi = offset * intput_padding + intput_offset;
		unsigned int oo = offset * results_padding + results_offset;

		results[oo] = input[oi] + change;

		offset += skip;
	}
}


char* mallocCharVec(vector<char*>* src)
{
	char* dst;
	checkCudaErrors(cudaMalloc(&dst, src->size() * ITEM_SIZE * sizeof(char)));
	char* hd = (char*)malloc(src->size() * ITEM_SIZE * sizeof(char));
	for (int i = 0; i < src->size(); i++)
		memcpy(&(hd[i * ITEM_SIZE]), src->at(i), ITEM_SIZE);
	checkCudaErrors(cudaMemcpy(dst, hd, src->size() * ITEM_SIZE * sizeof(char), cudaMemcpyHostToDevice));
	delete hd;
	return dst;
}


unsigned char* fileToCuda(const char* file_name, unsigned int* length)
{
	// Read file to RAM
	auto file = openFile(file_name);
	unsigned char* data = (unsigned char*)malloc(LINES * ITEM_SIZE);

	int lines = LINES;
	int sum = 0;
	while (lines >= LINES)
	{
		lines = readFile(file, data, LINES);
		sum += lines;
		if (lines >= LINES)
			data = (unsigned char*)realloc(data, sum * ITEM_SIZE);
	}

	closeFile(file);
	delete file;

	*length = sum;

	// Move from RAM to GPU
	unsigned char* cuda_array;
	checkCudaErrors(cudaMalloc((void**)&cuda_array, sum * ITEM_SIZE));
	checkCudaErrors(cudaMemcpy(cuda_array, data, sum * ITEM_SIZE, cudaMemcpyHostToDevice));

	// Free RAM
	free(data);

	return cuda_array;
}

void computeVectors()
{
	// Load neutral urls
	unsigned int invalid_length;
	unsigned char* d_invalid = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/urls_invalid_chars.txt", &invalid_length);

	// Load safe urls
	unsigned int safe_length;
	unsigned char* d_safe = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/safe_urls.txt", &safe_length);

	// Load neutral urls
	unsigned int neutral_length;
	unsigned char* d_neutral = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/neutral_urls.txt", &neutral_length);

	// create stream
	cudaStream_t streamA, streamB;
	checkCudaErrors(cudaStreamCreate(&streamA));
	checkCudaErrors(cudaStreamCreate(&streamB));

	// Host allocations
	unsigned char* dataA;
	checkCudaErrors(cudaHostAlloc((void**)&dataA, LINES * LINE_SIZE, cudaHostAllocDefault));
	unsigned char* dataB;
	checkCudaErrors(cudaHostAlloc((void**)&dataB, LINES * LINE_SIZE, cudaHostAllocDefault));

	RESULT_TYPE* h_dataA;
	checkCudaErrors(cudaHostAlloc((void**)&h_dataA, LINES * RESULT_VALUES * sizeof(RESULT_TYPE), cudaHostAllocDefault));
	RESULT_TYPE* h_dataB;
	checkCudaErrors(cudaHostAlloc((void**)&h_dataB, LINES * RESULT_VALUES * sizeof(RESULT_TYPE), cudaHostAllocDefault));

	// Cuda allocations
	unsigned char* d_linesA;
	unsigned char* d_linesB;
	checkCudaErrors(cudaMalloc((void**)&d_linesA, LINES * LINE_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_linesB, LINES * LINE_SIZE));

	// Cuda parsed data
	unsigned char* d_stringsA;
	unsigned int* d_lengthsA;

	checkCudaErrors(cudaMalloc((void**)&d_stringsA, LINES * LINE_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_lengthsA, LINES * LINE_PARTS * sizeof(unsigned int)));

	unsigned char* d_stringsB;
	unsigned int* d_lengthsB;

	checkCudaErrors(cudaMalloc((void**)&d_stringsB, LINES * LINE_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_lengthsB, LINES * LINE_PARTS * sizeof(unsigned int)));
	
	// Cuda results
	RESULT_TYPE* d_resultsA;
	RESULT_TYPE* d_resultsB;
	// Id, UserId,
	checkCudaErrors(cudaMalloc((void**)&d_resultsA, LINES * RESULT_VALUES * sizeof(RESULT_TYPE)));
	checkCudaErrors(cudaMalloc((void**)&d_resultsB, LINES * RESULT_VALUES * sizeof(RESULT_TYPE)));

	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto results = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/results.csv");

	// Skip/Add header
	skipFile(requests, 1);
	if (results->is_open())
	{
		string header = "Id,UserId,y,m,d,H,M,S,UrlLen,AgentLen,UrlDanger,UrlSafe,UrlNeutral,AgentLower,AgentUpper,AgentNumbers,AgentSpecial,AgentWhites\n";
		results->write(header.c_str(), header.size());
	}

	clock_t start = clock();

	// Stream FILE -> RAM -> GPU -> RAM -> FILE
	int lines = LINES;
	int linesA = 0;
	int linesB = 0;
	int dim = 0, dim2 = (LINES + BLOCK_DIM - 1) / BLOCK_DIM;
	int sum = 0;
	while (lines >= LINES)
	{
		// ----------------- Stream A ----------------------------------
		lines = readFile(requests, dataA, LINES);
		dim = (lines + BLOCK_DIM - 1) / BLOCK_DIM;

		/*checkCudaErrors(cudaStreamSynchronize(streamB));
		appendFile(results, h_dataB, linesB);*/

		checkCudaErrors(cudaStreamSynchronize(streamA));
		appendFile(results, h_dataA, linesA);

		cudaMemcpyAsync(d_linesA, dataA, lines * LINE_SIZE, cudaMemcpyHostToDevice);		// Copy lines to device

		clearArray<<<dim2, BLOCK_DIM, 0, streamA>>>(d_stringsA, LINES * LINE_SIZE);
		splitLine<<<dim, BLOCK_DIM, 0, streamA>>>(d_linesA, lines, d_stringsA, d_lengthsA);	// Split lines to items

		// Compute vectors
		stringToInt<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 0, lines, RESULT_VALUES, 0, d_resultsA);				// Id
		stringToInt<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 3, lines, RESULT_VALUES, 1, d_resultsA);				// User Id
		dateToInt<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 2, lines, RESULT_VALUES, 2, 3, 4, 5, 6, 7, d_resultsA);	// Date
		copyTo<<<dim, BLOCK_DIM, 0, streamA>>>(d_lengthsA, LINE_PARTS, 1, lines, -2, RESULT_VALUES, 8, d_resultsA);		// Url len
		copyTo<<<dim, BLOCK_DIM, 0, streamA>>>(d_lengthsA, LINE_PARTS, 5, lines, -2, RESULT_VALUES, 9, d_resultsA);		// Agent len

		findPattern<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 1, lines, d_invalid, invalid_length, RESULT_VALUES, 10, d_resultsA);	// Url Danger
		findPattern<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 1, lines, d_safe, safe_length, RESULT_VALUES, 11, d_resultsA);			// Url Safe
		findPattern<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 1, lines, d_neutral, neutral_length, RESULT_VALUES, 12, d_resultsA);	// Url neutral
		
		countChars<<<dim, BLOCK_DIM, 0, streamA>>>(d_stringsA, 5, lines, RESULT_VALUES, 13, 14, 15, 16, 17, d_resultsA); // Agent chars

		cudaMemcpyAsync(h_dataA, d_resultsA, lines * RESULT_VALUES * sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streamA);
		
		linesA = lines;
		sum += lines;

		//// ----------------- Stream B ----------------------------------
		lines = readFile(requests, dataB, LINES);
		dim = (lines + BLOCK_DIM - 1) / BLOCK_DIM;

		/*checkCudaErrors(cudaStreamSynchronize(streamA));
		appendFile(results, h_dataA, linesA);*/

		checkCudaErrors(cudaStreamSynchronize(streamB));
		appendFile(results, h_dataB, linesB);
		
		cudaMemcpyAsync(d_linesB, dataB, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamB);

		clearArray<<<dim2, BLOCK_DIM, 0, streamB>>>(d_stringsB, LINES * LINE_SIZE);
		splitLine<<<dim, BLOCK_DIM, 0, streamB>>>(d_linesB, lines, d_stringsB, d_lengthsB);	// Split lines to items

		// Compute vectors
		stringToInt<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 0, lines, RESULT_VALUES, 0, d_resultsB);				// Id
		stringToInt<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 3, lines, RESULT_VALUES, 1, d_resultsB);				// User Id
		dateToInt<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 2, lines, RESULT_VALUES, 2, 3, 4, 5, 6, 7, d_resultsB);	// Date
		copyTo<<<dim, BLOCK_DIM, 0, streamB>>>(d_lengthsB, LINE_PARTS, 1, lines, -2, RESULT_VALUES, 8, d_resultsB);		// Url len
		copyTo<<<dim, BLOCK_DIM, 0, streamB>>>(d_lengthsB, LINE_PARTS, 5, lines, -2, RESULT_VALUES, 9, d_resultsB);		// Agent len

		findPattern<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 1, lines, d_invalid, invalid_length, RESULT_VALUES, 10, d_resultsB);	// Url Danger
		findPattern<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 1, lines, d_safe, safe_length, RESULT_VALUES, 11, d_resultsB);			// Url Safe
		findPattern<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 1, lines, d_neutral, neutral_length, RESULT_VALUES, 12, d_resultsB);	// Url neutral

		countChars<<<dim, BLOCK_DIM, 0, streamB>>>(d_stringsB, 5, lines, RESULT_VALUES, 13, 14, 15, 16, 17, d_resultsB); // Agent chars

		cudaMemcpyAsync(h_dataB, d_resultsB, lines * RESULT_VALUES * sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streamB);
		
		linesB = lines;
		sum += lines;

		printf("\r%d", sum);
	}
	printf("\n");

	// Append last data
	/*checkCudaErrors(cudaStreamSynchronize(streamA));
	appendFile(results, h_dataA, linesA);
	checkCudaErrors(cudaStreamSynchronize(streamB));
	appendFile(results, h_dataB, linesB);*/

	closeFile(requests);
	delete requests;

	closeFile(results);
	delete results;

	cudaStreamSynchronize(streamA);
	cudaStreamSynchronize(streamB);
	cudaStreamDestroy(streamA);
	cudaStreamDestroy(streamB);

	printf("Done in %.2f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	printf("%d Lines readed\n", sum);

	cudaFree(d_linesA);
	cudaFree(d_stringsA);
	cudaFree(d_lengthsA);
	cudaFree(d_resultsA);

	cudaFree(d_linesB);
	cudaFree(d_stringsB);
	cudaFree(d_lengthsB);
	cudaFree(d_resultsB);

	cudaFree(d_invalid);
	cudaFree(d_safe);
	cudaFree(d_neutral);

	cudaFreeHost(dataA);
	cudaFreeHost(dataB);

	cudaFreeHost(h_dataA);
	cudaFreeHost(h_dataB);
}

void extractCPU(unsigned int part)
{
	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto results = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/extract.txt");

	clock_t start = clock();

	if (!requests->is_open())
		throw exception("Stream is not open!");
	if (!results->is_open())
		throw exception("Stream is not open!");

	int lines = LINES;
	int sum = 0;
	bool success = true;
	while (success)
	{
		string line;
		unsigned int lines = 0;
		for (unsigned int i = 0; i < LINES && success; i++)
		{
			success = !getline(*requests, line).eof();
			if (success)
			{
				auto parts = split(line, "\",\"");
				if (parts.size() > part)
				{
					line = parts[part] + "\n";
					results->write(line.c_str(), line.size());
				}
				lines++;
			}
		}
		sum += lines;
		printf("\r%d", sum);
	}

	printf("Done in %.2f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	printf("%d Lines readed\n", sum);

	closeFile(results);
	delete results;
	closeFile(requests);
	delete requests;

}

void readCPUBenchmark()
{
	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");

	unsigned char* data;
	cudaHostAlloc((void**)&data, LINES * LINE_SIZE, cudaHostAllocDefault);

	clock_t start = clock();
	
	int lines = LINES;
	int sum = 0;
	while (lines >= LINES)
	{
		lines = readFile(requests, data, LINES);
		sum += lines;
	}

	printf("Done in %.2f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	printf("%d Lines readed\n", sum);

	closeFile(requests);
	delete requests;

}

void readCopyCPUBenchmark()
{
	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");

	// create stream
	cudaStream_t streamA, streamB;
	cudaStreamCreate(&streamA);
	cudaStreamCreate(&streamB);

	unsigned char* dataA;
	cudaHostAlloc((void**)&dataA, LINES * LINE_SIZE, cudaHostAllocDefault);
	unsigned char* dataB;
	cudaHostAlloc((void**)&dataB, LINES * LINE_SIZE, cudaHostAllocDefault);

	unsigned char* d_linesA;
	cudaMalloc((void**)&d_linesA, LINE_SIZE);
	unsigned char* d_linesB;
	cudaMalloc((void**)&d_linesB, LINE_SIZE);

	clock_t start = clock();

	int lines = LINES;
	int sum = 0;
	while (lines >= LINES)
	{
		lines = readFile(requests, dataA, LINES);
		cudaMemcpyAsync(d_linesA, dataA, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamA);
		sum += lines;

		lines = readFile(requests, dataB, LINES);
		cudaMemcpyAsync(d_linesB, dataB, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamB);
		sum += lines;
	}

	cudaStreamSynchronize(streamA);
	cudaStreamSynchronize(streamB);
	cudaStreamDestroy(streamA);
	cudaStreamDestroy(streamB);

	printf("Done in %.2f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	printf("%d Lines readed\n", sum);

	cudaFree(d_linesA);
	cudaFree(d_linesB);

	cudaFreeHost(dataA);
	cudaFreeHost(dataB);

	closeFile(requests);
	delete requests;

}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	//readCPUBenchmark();
	//readCopyCPUBenchmark();
	computeVectors();
	//extractCPU(2);
}

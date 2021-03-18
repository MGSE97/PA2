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

#define BLOCK_DIM 1024
//#define LINES 4096
#define LINES 8196
#define ITEM_SIZE 4096
#define LINE_SIZE 7 * ITEM_SIZE
#define EMPTY_CHAR 205
#define RESULT_VALUES 5

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

void appendFile(ofstream* stream, int* data, const unsigned int lines)
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

__global__ void evalUrls(
	unsigned char* urls,
	unsigned int* urls_lenghts,
	unsigned char* safe,
	unsigned char* neutral,
	unsigned int urls_lenght,
	unsigned int safe_lenght,
	unsigned int neutral_lenght,
	int* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	//unsigned int skip = blockDim.x * gridDim.x;

	if(offset < urls_lenght)
	{
		char ok = -2;
		int o = offset * ITEM_SIZE;
		int len = urls_lenghts[offset];

		bool php = false, asp = false, amp = false;
		const char* sphp = ".php", * sasp = ".asp", * samp = "&";
		int pphp = 0, pasp = 0, pamp = 0;
		
		// dangerous chars
		for (int j = 0; j < len; j++) {
			char c = urls[o + j];

			php = sphp[pphp] == c;
			pphp = php ? pphp + 1 : 0;

			asp = sasp[pasp] == c;
			pasp = asp ? pasp + 1 : 0;

			amp = samp[pamp] == c;
			pamp = amp ? pamp + 1 : 0;

			if (pphp == 4 || pasp == 4 || pamp == 1)
				break;
			/*if (c <= 0)
			{
				php = asp = amp = false;
				break;
			}*/
		}

		php = pphp == 4;
		asp = pasp == 4;
		amp = pamp == 1;

		if (!(php || asp || amp))
		{
			ok = -1;

			// safe urls
			for (int i = 0; ok == -1 && i < safe_lenght; i++)
			{
				int off = i * ITEM_SIZE;
				bool eq = true;
				for (int j = 0; eq && j < len; j++)
					eq = safe[off + j] == urls[o + j];
				eq = eq && safe[off + len] == EMPTY_CHAR;

				ok = (eq ? 1 : -1);
			}

			// neutral urls
			for (int i = 0; ok == -1 && i < neutral_lenght; i++)
			{
				int off = i * ITEM_SIZE;
				bool eq = true;
				for (int j = 0; eq && j < len; j++)
					eq = neutral[off + j] == urls[o + j];
				eq = eq && neutral[off + len] == EMPTY_CHAR;

				ok = (eq ? 0 : -1);
			}
		}

		results[offset * RESULT_VALUES + 1] = ok;
		//offset += skip;
	}
}

__global__ void evalAgents(
	char* agents,
	const unsigned long agents_lenght,
	int* length)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	//unsigned int skip = blockDim.x * gridDim.x;

	if(offset < agents_lenght)
	{
		int o = offset * 4096;
		
		// todo

		//offset += skip;
	}
}

__global__ void prepareData(
	unsigned char* lines,
	const unsigned long lines_lenght,

	unsigned int* ids,
	unsigned char* urls,
	unsigned int* urls_length,
	tm* createds,
	unsigned int* user_ids,
	unsigned char* ips,
	unsigned int* ips_length,
	unsigned char* agents,
	unsigned int* agents_length,
	int* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	//unsigned int skip = blockDim.x * gridDim.x;

	if (offset < lines_lenght)
	{
		int ol = offset * LINE_SIZE;
		int oi = offset * ITEM_SIZE;
		int or = offset * RESULT_VALUES;
		char spliter = ',';
		char zero = '0';
		char nine = '9';
		char escape = '"';

		int val = 0;
		int started = 0;

		// Parse line
		unsigned int i = 0;
		unsigned char c = 0;
		unsigned char oc = 0;
		// get Id
		for (; i < LINE_SIZE && c != spliter; i++)
		{
			c = lines[ol + i];
			val = c >= zero && c <= nine ? val * 10 + c - zero : val;
		}
		ids[offset] = val;
		results[or ] = val;
		val = 0;
		c = 0;

		// get Url
		for (; i < LINE_SIZE && !(oc == escape && c == spliter); i++)
		{
			oc = c;
			c = lines[ol + i];
			if (started > 0 && c != spliter)
				urls[oi + i - started] = c;
			else if (c == escape)
				started = i+1;
		}
		/*for (unsigned int j = i - started - 2; j < ITEM_SIZE; j++)
			urls[oi + j] = EMPTY_CHAR;*/
		urls_length[offset] = i - started - 2;
		results[or + 2] = i - started - 2;
		c = 0;
		started = 0;

		// get Date
		for (; i < LINE_SIZE && !(oc == escape && c == spliter); i++)
		{
			oc = c;
			c = lines[ol + i];
			// ToDo
		}
		c = 0;

		// get User Id
		for (; i < LINE_SIZE && c != spliter; i++)
		{
			c = lines[ol + i];
			val = c >= zero && c <= nine ? val * 10 + c - zero : val;
		}
		user_ids[offset] = val;
		results[or + 3] = val;
		c = 0;

		// get Ip
		for (; i < LINE_SIZE && !(oc == escape && c == spliter); i++)
		{
			oc = c;
			c = lines[ol + i];
			if (started > 0 && c != spliter)
				ips[oi + i - started] = c;
			else if (c == escape)
				started = i + 1;
		}
		if (started == 0)
		{
			/*for (unsigned int j = 0; j < ITEM_SIZE; j++)
				ips[oi + j] = EMPTY_CHAR;*/
			ips_length[offset] = 0;
		}
		else
		{
			/*for (unsigned int j = i - started - 2; j < ITEM_SIZE; j++)
				ips[oi + j] = EMPTY_CHAR;*/
			ips_length[offset] = i - started - 2;
		}
		c = 0;
		started = 0;

		// get Agent
		for (; i < LINE_SIZE && !(oc == escape && c == spliter); i++)
		{
			oc = c;
			c = lines[ol + i];
			if (started > 0 && c != spliter)
				agents[oi + i - started] = c;
			else if (c == escape)
				started = i + 1;
		}
		if (started == 0)
		{
			/*for (unsigned int j = 0; j < ITEM_SIZE; j++)
				agents[oi + j] = EMPTY_CHAR;*/
			agents_length[offset] = 0;
			results[or + 4] = 0;
		}
		else
		{
			/*for (unsigned int j = i - started - 2; j < ITEM_SIZE; j++)
				agents[oi + j] = EMPTY_CHAR;*/
			agents_length[offset] = i - started - 2;
			results[or + 4] = i - started - 2;
		}

		//offset += skip;
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
	// Load safe urls
	unsigned int safe_length;
	unsigned char* d_safe = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/safe_urls.txt", &safe_length);
	//checkDeviceMatrix(d_safe, ITEM_SIZE, 20, 60, "%d ");

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

	int* h_dataA;
	checkCudaErrors(cudaHostAlloc((void**)&h_dataA, LINES * RESULT_VALUES * sizeof(int), cudaHostAllocDefault));
	int* h_dataB;
	checkCudaErrors(cudaHostAlloc((void**)&h_dataB, LINES * RESULT_VALUES * sizeof(int), cudaHostAllocDefault));

	// Cuda allocations
	unsigned char* d_linesA;
	unsigned char* d_linesB;
	checkCudaErrors(cudaMalloc((void**)&d_linesA, LINES * LINE_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_linesB, LINES * LINE_SIZE));
	
	// Cuda parsed data
	unsigned int* d_idsA;
	unsigned int* d_user_idsA;
	unsigned char* d_urlsA;
	unsigned char* d_ipsA;
	unsigned char* d_agentsA;
	unsigned int* d_urls_lengthA;
	unsigned int* d_ips_lengthA;
	unsigned int* d_agents_lengthA;
	tm* d_createdsA;
	checkCudaErrors(cudaMalloc((void**)&d_idsA, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_user_idsA, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_urlsA, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_ipsA, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_agentsA, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_createdsA, LINES * sizeof(tm)));
	checkCudaErrors(cudaMalloc((void**)&d_urls_lengthA, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_ips_lengthA, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_agents_lengthA, LINES * sizeof(unsigned int)));

	unsigned int* d_idsB;
	unsigned int* d_user_idsB;
	unsigned char* d_urlsB;
	unsigned char* d_ipsB;
	unsigned char* d_agentsB;
	unsigned int* d_urls_lengthB;
	unsigned int* d_ips_lengthB;
	unsigned int* d_agents_lengthB;
	tm* d_createdsB;
	checkCudaErrors(cudaMalloc((void**)&d_idsB, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_user_idsB, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_urlsB, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_ipsB, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_agentsB, LINES * ITEM_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_createdsB, LINES * sizeof(tm)));
	checkCudaErrors(cudaMalloc((void**)&d_urls_lengthB, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_ips_lengthB, LINES * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_agents_lengthB, LINES * sizeof(unsigned int)));

	// Cuda results
	int* d_resultsA;
	int* d_resultsB;
	// Id, Url Danger, Url len, UserId, Agent Length
	checkCudaErrors(cudaMalloc((void**)&d_resultsA, LINES * RESULT_VALUES * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_resultsB, LINES * RESULT_VALUES * sizeof(int)));

	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto results = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/results.csv");

	// Skip/Add header
	skipFile(requests, 1);
	if (results->is_open())
		results->write("Id,UrlDanger,UrlLen,UserId,AgentLen\n", 36);

	clock_t start = clock();

	// Stream FILE -> RAM -> GPU -> RAM -> FILE
	int lines = LINES;
	int linesA = 0;
	int linesB = 0;
	int dim = 0;
	int sum = 0;
	while (lines >= LINES)
	{
		// ----------------- Stream A ----------------------------------
		lines = readFile(requests, dataA, LINES);
		dim = (lines + BLOCK_DIM - 1) / BLOCK_DIM;

		checkCudaErrors(cudaStreamSynchronize(streamB));
		appendFile(results, h_dataB, linesB);

		/*cudaStreamSynchronize(streamA);
		appendFile(results, h_dataA, linesA);*/
		//checkDeviceMatrix(d_resultsA, RESULT_VALUES * sizeof(int), 10, 5, "%d ");
		//checkDeviceMatrix(dataA, LINE_SIZE * sizeof(char), 10, 40, "%c");
		//checkDeviceMatrix(d_idsA, sizeof(int), 10, 1, "%d");
		//checkHostMatrix(h_dataA, RESULT_VALUES * sizeof(int), 10, 5, "%d ");

		cudaMemcpyAsync(d_linesA, dataA, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamA);
		prepareData<<<dim, BLOCK_DIM, 0, streamA>>>(d_linesA, lines, d_idsA, d_urlsA, d_urls_lengthA, d_createdsA, d_user_idsA, d_ipsA, d_ips_lengthA, d_agentsA, d_agents_lengthA, d_resultsA);
		evalUrls<<<dim, BLOCK_DIM, 0, streamA>>>(d_urlsA, d_urls_lengthA, d_safe, d_neutral, lines, safe_length, neutral_length, d_resultsA);
		cudaMemcpyAsync(h_dataA, d_resultsA, lines * RESULT_VALUES * sizeof(int), cudaMemcpyDeviceToHost, streamA);
		
		linesA = lines;
		sum += lines;

		//// ----------------- Stream B ----------------------------------
		lines = readFile(requests, dataB, LINES);
		dim = (lines + BLOCK_DIM - 1) / BLOCK_DIM;

		checkCudaErrors(cudaStreamSynchronize(streamA));
		appendFile(results, h_dataA, linesA);

		/*checkCudaErrors(cudaStreamSynchronize(streamB));
		appendFile(results, h_dataB, linesB);*/
		
		checkCudaErrors(cudaMemcpyAsync(d_linesB, dataB, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamB));
		prepareData<<<dim, BLOCK_DIM, 0, streamB>>>(d_linesB, lines, d_idsB, d_urlsB, d_urls_lengthB, d_createdsB, d_user_idsB, d_ipsB, d_ips_lengthB, d_agentsB, d_agents_lengthB, d_resultsB);
		evalUrls<<<dim, BLOCK_DIM, 0, streamB>>>(d_urlsB, d_urls_lengthB, d_safe, d_neutral, lines, safe_length, neutral_length, d_resultsB);
		checkCudaErrors(cudaMemcpyAsync(h_dataB, d_resultsB, lines * RESULT_VALUES * sizeof(int), cudaMemcpyDeviceToHost, streamB));
		
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
	cudaFree(d_linesB);


	cudaFree(d_idsA);
	cudaFree(d_user_idsA);
	cudaFree(d_urlsA);
	cudaFree(d_ipsA);
	cudaFree(d_agentsA);
	cudaFree(d_urls_lengthA);
	cudaFree(d_ips_lengthA);
	cudaFree(d_agents_lengthA);
	cudaFree(d_createdsA);

	cudaFree(d_idsB);
	cudaFree(d_user_idsB);
	cudaFree(d_urlsB);
	cudaFree(d_ipsB);
	cudaFree(d_agentsB);
	cudaFree(d_urls_lengthB);
	cudaFree(d_ips_lengthB);
	cudaFree(d_agents_lengthB);
	cudaFree(d_createdsB);

	cudaFreeHost(dataA);
	cudaFreeHost(dataB);

	cudaFreeHost(h_dataA);
	cudaFreeHost(h_dataB);
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
}

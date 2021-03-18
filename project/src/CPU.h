#pragma once
#include <string>
#include <vector>
#include "File.h"
#include <ctime>

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
	unsigned int len = 0;
	while (lines >= LINES)
	{
		lines = readFile(requests, data, LINES, LINE_SIZE, &len);
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
	unsigned int len = 0;
	while (lines >= LINES)
	{
		lines = readFile(requests, dataA, LINES, LINE_SIZE, &len);
		cudaMemcpyAsync(d_linesA, dataA, lines * LINE_SIZE, cudaMemcpyHostToDevice, streamA);
		sum += lines;

		lines = readFile(requests, dataB, LINES, LINE_SIZE, &len);
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
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Settings.h"
#include <cudaDefs.h>

using namespace std;

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

unsigned int readFile(ifstream* stream, unsigned char* data, const unsigned int lines, const unsigned int padding, unsigned int* max_len)
{
	if (!stream->is_open())
		throw exception("Stream is not open!");

	string line;
	bool success = true;
	unsigned int len = 0;
	*max_len = 0;
	for (unsigned int i = 0; i < lines && success; i++)
	{
		success = !getline(*stream, line).eof();
		if (success)
		{
			memcpy(&data[i * padding], line.c_str(), line.size() * sizeof(char));
			if (*max_len < line.size())
				*max_len = MINIMUM(line.size(), LINE_SIZE);
			len++;
		}
	}
	return len;
}

void appendFile(ofstream* stream, RESULT_TYPE* data, unsigned char* texts, const unsigned int lines)
{
	if (lines == 0)
		return;

	if (!stream->is_open())
		throw exception("Stream is not open!");

	string write;
	for (unsigned int i = 0; i < lines; i++)
	{
		unsigned int o = i * RESULT_VALUES;
		for (unsigned int j = 0; j < RESULT_VALUES; j++)
		{
			write += to_string(data[o + j]);
			write.push_back(',');
		}
		o = i * TEXT_RESULT_VALUES_SIZE;
		for (unsigned int j = 0; j < TEXT_RESULT_VALUES; j++)
		{
			for (unsigned int k = 1; k < ITEM_SIZE; k++)
			{
				const unsigned char c = texts[o + k];
				if (c == ',' || c == EMPTY_CHAR || c == 0)
					break;
				else
					write.push_back(c);
			}
			write.pop_back();
			write.push_back(',');
			o += ITEM_SIZE;
		}
		write.pop_back();
		write.push_back('\n');
	}
	stream->write(write.c_str(), write.size());
}

void clasifyFile(ofstream* stream, RESULT_TYPE* data, const unsigned int lines, const unsigned int result_offset)
{
	if (lines == 0)
		return;

	if (!stream->is_open())
		throw exception("Stream is not open!");

	string labels[] = { "neutral", "user", "dangerous", "search_engine", "trusted", "guest" };
	string write;
	unsigned int o = 0;
	for (unsigned int i = 0; i < lines; i++)
	{
		write += to_string(data[o]); // id
		write.push_back(',');
		write += labels[data[o + result_offset]];
		write.push_back('\n');
		o += RESULT_VALUES;
	}
	stream->write(write.c_str(), write.size());
}

unsigned char* fileToCuda(const char* file_name, unsigned int* length, unsigned int* max_len)
{
	// Read file to RAM
	auto file = openFile(file_name);
	unsigned char* data = (unsigned char*)malloc(LINES * ITEM_SIZE);

	int lines = LINES;
	int sum = 0;
	unsigned int len = 0;
	*max_len = 0;
	while (lines >= LINES)
	{
		lines = readFile(file, data, LINES, ITEM_SIZE, &len);
		sum += lines;
		if (lines >= LINES)
			data = (unsigned char*)realloc(data, sum * ITEM_SIZE);
		if (len > *max_len)
			*max_len = len;
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
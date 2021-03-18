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
#define LINES 4096
#define ITEM_SIZE 4096
#define LINE_SIZE 7 * ITEM_SIZE;

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
	stream->open(file, ofstream::trunc); // Create
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

unsigned int readFile(ifstream* stream, vector<string>* data, const unsigned int lines)
{
	if (!stream->is_open())
		throw exception("Stream is not open!");

	string line;
	bool success = true;
	unsigned int i = 0;
	for (; i < lines && success; i++)
	{
		success = !getline(*stream, line).eof();
		if (success)
			data->push_back(line);
	}
	return i;
}

void appendFile(ofstream* stream, vector<string>* data)
{
	if (!stream->is_open())
		throw exception("Stream is not open!");

	for (string line : *data)
	{
		stream->write((line + "\n").c_str(), line.size() + 1);
	}
}

void convertLineToData(string* line, char* data_url, string* data_id, tm* data_created_utc, char* data_ip_address, char* data_user_agent, string* data_user_id)
{
	auto item = split(*line, ","); // Fix ,",",
	if (item.size() > 1)
	{
		// get id
		*data_id = item.at(0);

		// get url
		auto url = item.at(1);

		if (url.size() > 3)
			for(int i = 0;i < url.size()-2; i++)
				data_url[i] = url[i+1];

		// get created utc
		auto created = item.at(2);
		
		tm dt;
		sscanf(created.c_str(), "\"%4d-%2d-%2d %2d:%2d:%2d\"", &dt.tm_year, &dt.tm_mon, &dt.tm_mday, &dt.tm_hour, &dt.tm_min, &dt.tm_sec);
		dt.tm_mon--;
		*data_created_utc = dt;

		// get user id ?
		*data_user_id = item.at(3) == "\\N" ? "-1" : item.at(3);

		// get ip address ?
		auto ip = item.at(4);

		if(ip.size() > 3)
			for (int i = 0; i < ip.size()-2; i++)
				data_ip_address[i] = ip[i+1];

		// get user agent ?
		auto user_agent = item.at(5);
		if (user_agent.size() > 3)
			for (int i = 0; i < user_agent.size()-2; i++)
				data_user_agent[i] = user_agent[i+1];

		// get request ?
		// ignore
	}
}

void getUrlCPU()
{
	// ToDo smth
	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto urls_file = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/urls.txt");

	vector<string>* urls = new vector<string>();

	vector<string>* data = new vector<string>();
	int lines = LINES, sum = 0;
	while (lines >= LINES)
	{
		lines = readFile(requests, data, LINES);
		for (int i = 0; i < data->size(); i++)
		{
			//printf("%s\n", data->at(i).c_str());
			auto item = split(data->at(i), ";");
			if (item.size() > 1)
			{
				auto url = item.at(1);
				url = url.substr(1, url.size() - 2);
				if (find(urls->begin(), urls->end(), url) == urls->end())
					urls->push_back(url);
			}
		}
		sum += lines;
		printf("%d\n", sum);
		data->clear();
	}

	appendFile(urls_file, urls);
	delete urls;

	closeFile(requests);
	delete requests;

	closeFile(urls_file);
	delete urls_file;
}

__global__ void evalUrls(
	char* urls,
	char* safe,
	char* neutral,
	const unsigned long urls_lenght,
	const unsigned long safe_lenght,
	const unsigned long neutral_lenght,
	char* danger,
	int* length)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = blockDim.x * gridDim.x;

	while (offset < urls_lenght)
	{
		char ok = -1;
		int o = offset * 4096;
		int len = 0;

		bool php = false, asp = false, amp = false;
		const char* sphp = ".php", * sasp = ".asp", * samp = "&";
		int pphp = 0, pasp = 0, pamp = 0;

		// lenght
		for (; len < 4096 && urls[o + len] > 0; len++);
			

		// dangerous chars
		for (int j = 0; j < 4096; j++) {
			char c = urls[o + j];

			php = sphp[pphp] == c;
			pphp = php ? pphp + 1 : 0;

			asp = sasp[pasp] == c;
			pasp = asp ? pasp + 1 : 0;

			amp = samp[pamp] == c;
			pamp = amp ? pamp + 1 : 0;

			if (pphp == 4 || pasp == 4 || pamp == 1)
				break;
			if (c <= 0)
			{
				php = asp = amp = false;
				break;
			}
		}

		if (!(php || asp || amp))
		{
			// safe urls
			for (int i = 0; ok == -1 && i < safe_lenght; i++)
			{
				int off = i * 4096;
				bool eq = true;
				for (int j = 0; eq && j < 4096; j++)
					eq = safe[off + j] == urls[o + j];

				ok = (eq ? 1 : -1);
			}

			// neutral urls
			for (int i = 0; ok == -1 && i < neutral_lenght; i++)
			{
				int off = i * 4096;
				bool eq = true;
				for (int j = 0; eq && j < 4096; j++)
					eq = neutral[off + j] == urls[o + j];

				ok = (eq ? 0 : -1);
			}
		}

		length[offset] = len;
		danger[offset] = ok;
		offset += skip;
	}
}

__global__ void evalAgents(
	char* agents,
	const unsigned long agents_lenght,
	int* length)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = blockDim.x * gridDim.x;

	while (offset < agents_lenght)
	{
		int o = offset * 4096;
		int len = 0;

		// length
		for (; len < 4096 && agents[o + len] > 0; len++);

		// todo

		length[offset] = len;
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

void categorizeURLs()
{
	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto urls_safe_file = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/safe_urls.txt");
	auto urls_neutral_file = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/neutral_urls.txt");
	auto urls_results_file = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/results.csv");

	// Load urls categorization
	vector<char*>* urls_safe = new vector<char*>();
	vector<char*>* urls_neutral = new vector<char*>();

	vector<string>* data = new vector<string>();

	// Load safe urls
	int lines = LINES;
	while (lines >= LINES)
	{
		lines = readFile(urls_safe_file, data, LINES);
		for (int i = 0; i < data->size(); i++)
		{
			char* url = (char*)malloc(ITEM_SIZE*sizeof(char));
			memcpy(url, data->at(i).c_str(), data->at(i).size() * sizeof(char));
			if (find(urls_safe->begin(), urls_safe->end(), url) == urls_safe->end())
				urls_safe->push_back(url);
		}
		data->clear();
	}

	// Load neutral urls
	lines = LINES;
	while (lines >= LINES)
	{
		lines = readFile(urls_neutral_file, data, LINES);
		for (int i = 0; i < data->size(); i++)
		{
			char* url = (char*)malloc(ITEM_SIZE * sizeof(char));
			memcpy(url, data->at(i).c_str(), data->at(i).size() * sizeof(char));
			if (find(urls_neutral->begin(), urls_neutral->end(), url) == urls_neutral->end())
				urls_neutral->push_back(url);
		}
		data->clear();
	}

	// Write header
	data->push_back("Id,UrlDanger,UrlLength,UserId,AgentLength");
	appendFile(urls_results_file, data);
	data->clear();
	
	lines = LINES;
	int sum = 0;
	bool first = true;
	
	// Prepare row values
	auto urls = new vector<char*>();
	auto agents = new vector<char*>();
	auto ips = new vector<char*>();
	auto dates = new vector<tm*>();
	
	auto ids = new vector<string>();
	auto ids_old = new vector<string>();

	auto user_ids = new vector<string>();
	auto user_ids_old = new vector<string>();

	string id;
	string user_id;

	int urls_size = 0;
	char* urls_safe_data, * urls_neutral_data;
	urls_safe_data = mallocCharVec(urls_safe);
	urls_neutral_data = mallocCharVec(urls_neutral);
	
	clock_t start = clock();

	// Load Categorize and Save results
	while (lines > 0)
	{
		// Load data
		if (first)
		{
			data->clear();
			lines = readFile(requests, data, LINES);
			for (int i = 0; i < data->size(); i++)
			{
				char* url = (char*)malloc(ITEM_SIZE * sizeof(char));
				char* agent = (char*)malloc(ITEM_SIZE * sizeof(char));
				char* ip = (char*)malloc(ITEM_SIZE * sizeof(char));
				tm* date = (tm*)malloc(sizeof(tm));
				convertLineToData(&(data->at(i)), url, &id, date, ip, agent, &user_id);
				if (id == "Id" || id == "id")
				{
					SAFE_DELETE(url);
					SAFE_DELETE(agent);
					SAFE_DELETE(ip);
					SAFE_DELETE(date);
				}
				else
				{
					urls->push_back(url);
					ids->push_back(id);
					user_ids->push_back(user_id);
					agents->push_back(agent);
					ips->push_back(ip);
					dates->push_back(date);
				}
			}
			first = false;
			sum += lines;
		}

		// Process data
		char* urls_data;
		char* urls_dangers;
		int* urls_lengths;

		char* agents_data;
		int* agents_lengths;

		urls_size = urls->size();
		checkCudaErrors(cudaMalloc(&urls_dangers, urls_size * sizeof(char)));
		checkCudaErrors(cudaMalloc(&urls_lengths, urls_size * sizeof(int)));

		checkCudaErrors(cudaMalloc(&agents_lengths, urls_size * sizeof(int)));

		urls_data = mallocCharVec(urls);
		agents_data = mallocCharVec(agents);

		KernelSetting ks;
		ks.dimBlock = dim3(BLOCK_DIM, 1, 1);
		ks.dimGrid = dim3(MINIMUM(getNumberOfParts(urls_size, BLOCK_DIM), 64), 1, 1);

		evalUrls<<<ks.dimGrid, ks.dimBlock>>>(urls_data, urls_safe_data, urls_neutral_data, urls_size, urls_safe->size(), urls_neutral->size(), urls_dangers, urls_lengths);
		evalAgents<<<ks.dimGrid, ks.dimBlock>>>(agents_data, urls_size, agents_lengths);

		// Clean up async
		for (auto x : *urls)
			SAFE_DELETE(x);
		for (auto x : *agents)
			SAFE_DELETE(x);
		for (auto x : *ips)
			SAFE_DELETE(x);
		for (auto x : *dates)
			SAFE_DELETE(x);
		ids_old->clear();
		for (auto id : *ids)
			ids_old->push_back(id);
		user_ids_old->clear();
		for (auto user_id : *user_ids)
			user_ids_old->push_back(user_id);

		// Load next data async
		if (lines >= LINES)
		{
			data->clear();
			lines = readFile(requests, data, LINES);
			urls->clear();
			ids->clear();
			user_ids->clear();
			agents->clear();
			ips->clear();
			dates->clear();
			for (int i = 0; i < data->size(); i++)
			{
				char* url = (char*)malloc(ITEM_SIZE * sizeof(char));
				char* agent = (char*)malloc(ITEM_SIZE * sizeof(char));
				char* ip = (char*)malloc(ITEM_SIZE * sizeof(char));
				tm* date = (tm*)malloc(sizeof(tm));
				convertLineToData(&(data->at(i)), url, &id, date, ip, agent, &user_id);
				urls->push_back(url);
				ids->push_back(id);
				user_ids->push_back(user_id);
				agents->push_back(agent);
				ips->push_back(ip);
				dates->push_back(date);
			}

			sum += lines;
		}
		else
			lines = 0;

		// Wait for kernels to finish
		cudaDeviceSynchronize();
		auto ex = cudaGetLastError();
		if (ex != NULL)
			printf("Error: %s\n", cudaGetErrorString(ex));

		// Save results
		char* hurl_dangers = (char*)malloc(urls_size * sizeof(char));
		checkCudaErrors(cudaMemcpy(hurl_dangers, urls_dangers, urls_size * sizeof(char), cudaMemcpyDeviceToHost));
		int* hurl_lengths = (int*)malloc(urls_size * sizeof(int));
		checkCudaErrors(cudaMemcpy(hurl_lengths, urls_lengths, urls_size * sizeof(int), cudaMemcpyDeviceToHost));

		int* hagent_lengths = (int*)malloc(urls_size * sizeof(int));
		checkCudaErrors(cudaMemcpy(hagent_lengths, agents_lengths, urls_size * sizeof(int), cudaMemcpyDeviceToHost));

		data->clear();
		for (int i = 0; i < urls_size; i++)
			data->push_back(
				ids_old->at(i) + "," + 
				to_string(hurl_dangers[i]) + "," + 
				to_string(hurl_lengths[i]) + "," + 
				user_ids_old->at(i) + "," +
				to_string(hagent_lengths[i]) + ",");

		appendFile(urls_results_file, data);

		// Clean up
		SAFE_DELETE_CUDA(urls_data);
		SAFE_DELETE_CUDA(urls_dangers);
		SAFE_DELETE_CUDA(urls_lengths);

		SAFE_DELETE_CUDA(agents_data);
		SAFE_DELETE_CUDA(agents_lengths);
		
		SAFE_DELETE(hurl_dangers);
		SAFE_DELETE(hurl_lengths);

		SAFE_DELETE(hagent_lengths);

		printf("%d\n", sum);
	}

	printf("Done in %.2f s", (std::clock() - start) / (double)CLOCKS_PER_SEC);

	SAFE_DELETE_CUDA(urls_neutral_data);
	SAFE_DELETE_CUDA(urls_safe_data);

	ids_old->clear();
	user_ids_old->clear();

	closeFile(urls_results_file);
	delete urls_results_file;

	closeFile(requests);
	delete requests;

	closeFile(urls_safe_file);
	delete urls_safe_file;
	delete urls_safe;

	closeFile(urls_neutral_file);
	delete urls_neutral_file;
	delete urls_neutral;

}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	categorizeURLs();
}

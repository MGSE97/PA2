#include "Runner.h"

cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

void computeVectors()
{
	// Load neutral urls
	unsigned int invalid_length;
	unsigned int invalid_item_length;
	unsigned char* d_invalid = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/urls_invalid_chars.txt", &invalid_length, &invalid_item_length);

	// Load safe urls
	unsigned int safe_length;
	unsigned int safe_item_length;
	unsigned char* d_safe = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/safe_urls.txt", &safe_length, &safe_item_length);

	// Load neutral urls
	unsigned int neutral_length;
	unsigned int neutral_item_length;
	unsigned char* d_neutral = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/neutral_urls.txt", &neutral_length, &neutral_item_length);

	// Load trusted agents
	unsigned int trusted_length;
	unsigned int trusted_item_length;
	unsigned char* d_trusted = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/agents_trusted.txt", &trusted_length, &trusted_item_length);

	// Load search agents
	unsigned int search_length;
	unsigned int search_item_length;
	unsigned char* d_search = fileToCuda("D:/Documents/Projekty/Škola/PA2/project/assets/agents_search.txt", &search_length, &search_item_length);

	// Create streams
	StreamData* streams = new StreamData[STREAMS];
	for (unsigned int i = 0; i < STREAMS; i++)
		CreateStream(&streams[i]);

	auto requests = openFile("D:/Documents/Projekty/Škola/PA2/project/assets/requestlog.csv");
	auto results = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/results.csv");
	auto clasifications = writeFile("D:/Documents/Projekty/Škola/PA2/project/assets/classifications.csv");

	// Skip/Add header
	skipFile(requests, 1);
	if (results->is_open())
	{
		string header = "Id,y,m,d,H,M,S,UrlDanger,UrlSafe,UrlNeutral,AgentTrusted,AgentSE,Classification,UserId,UrlLen,AgentLen,AgentLower,AgentUpper,AgentNumbers,AgentSpecial,AgentWhites,Url,Agent\n";
		results->write(header.c_str(), header.size());
	}
	
	// Stream FILE -> RAM -> GPU -> RAM -> FILE
	unsigned int lines = LINES;
	unsigned int sum = 0;
	unsigned int dim = (LINES + BLOCK_DIM - 1) / BLOCK_DIM;
	unsigned int items[LINE_PARTS];
	for (unsigned int i = 0; i < LINE_PARTS; i++)
		items[i] = i * ITEM_SIZE;

	unsigned char* data;
	checkCudaErrors(cudaHostAlloc((void**)&data, LINES * LINE_SIZE, cudaHostAllocDefault));

	clock_t start = clock();

	while (lines >= LINES)
	{
		for (unsigned int i = 0; i < STREAMS; i++)
			RunStream(lines, data, dim, requests, results, clasifications, &streams[i], sum, items, 
				d_safe, safe_length, d_neutral, neutral_length, d_invalid, invalid_length, 
				d_trusted, trusted_length, d_search, search_length);
	}
	printf("\n");

	printf("Done in %.2f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	printf("%d Lines readed\n", sum);
	
	closeFile(requests);
	delete requests;

	closeFile(results);
	delete results;

	closeFile(clasifications);
	delete clasifications;

	// Dispose streams
	for (unsigned int i = 0; i < STREAMS; i++)
		DisposeStream(&streams[i]);
	
	delete streams;

	cudaFree(d_invalid);
	cudaFree(d_safe);
	cudaFree(d_neutral);
	cudaFree(d_trusted);
	cudaFree(d_search);
}

void CreateStream(StreamData* data)
{
	// create stream
	checkCudaErrors(cudaStreamCreate(&data->stream));

	// Host allocations
	checkCudaErrors(cudaHostAlloc((void**)&data->h_data, LINES * RESULT_VALUES * sizeof(RESULT_TYPE), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&data->h_texts, LINES * TEXT_RESULT_VALUES_SIZE * sizeof(unsigned char), cudaHostAllocDefault));

	// Cuda allocations
	checkCudaErrors(cudaMalloc((void**)&data->d_lines, LINES * LINE_SIZE));

	// Cuda parsed data
	checkCudaErrors(cudaMalloc((void**)&data->d_strings, LINES * LINE_SIZE));
	checkCudaErrors(cudaMalloc((void**)&data->d_lengths, LINES * LINE_PARTS * sizeof(unsigned int)));

	// Cuda results
	checkCudaErrors(cudaMalloc((void**)&data->d_results, LINES * RESULT_VALUES * sizeof(RESULT_TYPE)));
	checkCudaErrors(cudaMalloc((void**)&data->d_text_results, LINES * TEXT_RESULT_VALUES_SIZE * sizeof(unsigned char)));

	data->h_lines = 0;
}

void DisposeStream(StreamData* data)
{
	cudaStreamSynchronize(data->stream);
	cudaStreamDestroy(data->stream);
	
	cudaFree(data->d_lines);
	cudaFree(data->d_strings);
	cudaFree(data->d_lengths);
	cudaFree(data->d_results);
	cudaFree(data->d_text_results);

	cudaFreeHost(data->data);
	cudaFreeHost(data->h_data);
	cudaFreeHost(data->h_texts);
}

void RunStream(unsigned int& lines, unsigned char* file_data, unsigned int& full_dim, std::ifstream* requests, std::ofstream* results, std::ofstream* clasifications, StreamData* data, unsigned int& sum, unsigned int* items,
	unsigned char* d_safe, unsigned int& safe_length, unsigned char* d_neutral, unsigned int& neutral_length, unsigned char* d_invalid, unsigned int& invalid_length, 
	unsigned char* d_trusted, unsigned int& trusted_length, unsigned char* d_search, unsigned int& search_length)
{
	unsigned int line_size, search_size, dim;
	lines = readFile(requests, file_data, LINES, LINE_SIZE, &line_size);
	search_size = MINIMUM(line_size, ITEM_SIZE);
	dim = (lines + BLOCK_DIM - 1) / BLOCK_DIM;
	
	checkCudaErrors(cudaStreamSynchronize(data->stream));
	appendFile(results, data->h_data, data->h_texts, data->h_lines);
	clasifyFile(clasifications, data->h_data, data->h_lines, 12);

	data->h_lines = lines;
	cudaMemcpyAsync(data->d_lines, file_data, data->h_lines * LINE_SIZE, cudaMemcpyHostToDevice, data->stream);		// Copy lines to device
	checkCudaErrors(cudaStreamSynchronize(data->stream));

	clearArray<<<full_dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, LINES * LINE_SIZE);
	splitLine<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_lines, data->h_lines, line_size, data->d_strings, data->d_lengths);	// Split lines to items

	// Compute vectors
	// Context data
	stringToInt<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[0], data->h_lines, search_size, 0, data->d_results);	// Id
	dateToInt<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[2], data->h_lines, 1, data->d_results);					// Date
	
	// Classificator data
	findPattern<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[1], data->h_lines, search_size, d_invalid, invalid_length, 7, data->d_results);	// Url Danger
	findPattern<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[1], data->h_lines, search_size, d_safe, safe_length, 8, data->d_results);		// Url Safe
	findPattern<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[1], data->h_lines, search_size, d_neutral, neutral_length, 9, data->d_results);	// Url neutral

	findPattern<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[5], data->h_lines, search_size, d_trusted, trusted_length, 10, data->d_results);	// Agent trusted
	findPattern<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[5], data->h_lines, search_size, d_search, search_length, 11, data->d_results);		// Agent search engine
	
	// Clasify
	clasify<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_results, data->h_lines, 7, 5);

	// Training data
	stringToInt<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[3], data->h_lines, search_size, 13, data->d_results);		// User Id
	copyTo<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_lengths, LINE_PARTS, 1, data->h_lines, -2, RESULT_VALUES, 14, data->d_results);	// Url len
	copyTo<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_lengths, LINE_PARTS, 5, data->h_lines, -2, RESULT_VALUES, 15, data->d_results);	// Agent len
	countChars<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[5], data->h_lines, search_size, 16, data->d_results);		// Agent chars
	sanitize<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[1], data->h_lines, search_size, 0, data->d_text_results);		// Sanitized Url
	sanitize<<<dim, BLOCK_DIM, 0, data->stream>>>(data->d_strings, items[5], data->h_lines, search_size, ITEM_SIZE, data->d_text_results); // Sanitized Agent


	cudaMemcpyAsync(data->h_data, data->d_results, data->h_lines * RESULT_VALUES * sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, data->stream);
	cudaMemcpyAsync(data->h_texts, data->d_text_results, data->h_lines * TEXT_RESULT_VALUES_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost, data->stream);

	sum += lines;
	printf("\r%d", sum);
}



int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	//readCPUBenchmark();
	//readCopyCPUBenchmark();
	//extractCPU(2);

	computeVectors();
}

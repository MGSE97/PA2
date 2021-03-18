#include "Vectorization.cuh"

//__constant__ unsigned int LineSize;
//__constant__ unsigned int SearchLength;
//
//void setLineSize(unsigned int line_size, cudaStream_t& stream) {
//	cudaMemcpyToSymbolAsync((const void*)&LineSize, (const void*)&line_size, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
//}
//
//void setSearchLength(unsigned int search_length, cudaStream_t& stream) {
//	cudaMemcpyToSymbolAsync((const void*)&SearchLength, (const void*)&search_length, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
//}

__global__ void clearArray(
	unsigned char* arr,
	const unsigned int lenght)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = gridDim.x * blockDim.x;

	while (offset < lenght)
	{
		arr[offset] = 0;
		offset += skip;
	}
}

__global__ void splitLine(
	unsigned char* __restrict__ lines,
	const unsigned int  lines_lenght,
	const unsigned int  line_size,
	unsigned char* strings,
	unsigned int* lengths)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (offset < lines_lenght)
	{
		unsigned int ol = offset * LINE_SIZE;
		unsigned int op = offset * LINE_PARTS;

		const unsigned char split = ',', escape = '\"';

		unsigned int part = 0;
		unsigned int len = 0;
		unsigned int pos = 0;

		unsigned char c_old = 0;
		bool escaped = false;
		bool do_split = false;
		for (unsigned int i = 0; i < line_size && pos < LINE_SIZE; i++)
		{
			const unsigned char c = lines[ol + i];

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
	const unsigned int strings_lenght,
	const unsigned int  search_size,
	unsigned char* __restrict__ patterns,
	const unsigned int patterns_lenght,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset_string = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset_string < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset_string * LINE_SIZE + part;

		unsigned int result = 0;
		for (unsigned int pattern = 0; pattern < patterns_lenght; pattern++)
		{
			unsigned int op = pattern * ITEM_SIZE;

			unsigned int p = 0;
			bool found = false;
			for (unsigned int i = 0; i < search_size; i++)
			{
				const unsigned char a = strings[os + i];
				const unsigned char b = patterns[op + p];

				found += (b == EMPTY_CHAR) ? (p > 0) : found;
				p = (a == b) ? (p + 1) : 0;
			}

			result += found;
		}

		results[offset_string * RESULT_VALUES + results_offset] = result > 0;
	}
}

__global__ void countChars(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int  search_size,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset_string = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset_string < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset_string * LINE_SIZE + part;

		const unsigned char whites_c[] = { ' ', '\f', '\n', '\r', '\t', '\v' };

		RESULT_TYPE counts[5];

		for (unsigned int i = 0; i < search_size; i++)
		{
			const unsigned char c = strings[os + i];

			counts[0] += (c >= 'a' && c <= 'z') ? 1 : 0;
			counts[1] += (c >= 'A' && c <= 'Z') ? 1 : 0;
			counts[2] += (c >= '0' && c <= '9') ? 1 : 0;

			bool is_white = false;
			for (unsigned int w = 0; w < 6; w++) {
				counts[3] += c == whites_c[w] ? 1 : 0;
				is_white = c == whites_c[w] ? true : is_white;
			}

			bool is_not_special = is_white;
			is_not_special = (c >= 'a' && c <= 'z') ? true : is_not_special;
			is_not_special = (c >= 'A' && c <= 'Z') ? true : is_not_special;
			is_not_special = (c >= '0' && c <= '9') ? true : is_not_special;
			is_not_special = c == EMPTY_CHAR ? true : is_not_special;
			is_not_special = c == 0 ? true : is_not_special;

			counts[4] += is_not_special ? 0 : 1;
		}

		os = offset_string * RESULT_VALUES + results_offset;
		for (unsigned int i = 0; i < 5; i++)
			results[os + i] = counts[i];
	}
}

__global__ void stringToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int search_size,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset * LINE_SIZE + part;

		const unsigned char zero = '0';
		const unsigned char nine = '9';

		RESULT_TYPE val = 0;

		// get int
		for (unsigned int i = 0; i < search_size; i++)
		{
			const unsigned char c = strings[os + i];
			val = (c >= zero && c <= nine) ? (val * 10 + c - zero) : val;
		}

		results[offset * RESULT_VALUES + results_offset] = val;
	}
}

__global__ void dateToInt(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int results_offset,
	RESULT_TYPE* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset * LINE_SIZE + part;

		const unsigned char zero = '0';
		const unsigned char nine = '9';

		RESULT_TYPE values[8];
		int type = -1;

		// get int
		for (unsigned int i = 0; i < 20; i++)
		{
			const unsigned char c = strings[os + i];
			bool number = c >= zero && c <= nine;
			type += number ? 0 : 1;
			values[type] = number ? (values[type] * 10 + c - zero) : 0;
		}

		os = offset * RESULT_VALUES + results_offset;
		for (unsigned int i = 0; i < 6; i++)
			results[os + i] = values[i];
	}
}

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

__global__ void clasify(
	RESULT_TYPE* results,
	const unsigned int lines,
	const unsigned int classificator_offset,
	const unsigned int resultS_offset) //relative
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < lines)
	{
		unsigned int or = offset * RESULT_VALUES + classificator_offset;

		RESULT_TYPE category = 0; // neutral

		category = (results[or + 5] > 0) ? 1 : 0; // user
		
		bool dangerous = results[or] > 0;
		category = dangerous ? 2 : category;	// dangerous url

		if (!dangerous)
		{
			bool safe = results[or + 1] > 0;	// safe url
			bool trusted = results[or + 3] > 0; // agent trusted

			category = safe ? 5 : category;						// guest
			category = safe && trusted ? 4 : category;			// trusted
			category = (results[or + 3] > 0) ? 3 : category;	// Search engine
		}

		results[or + resultS_offset] = category;
	}
}


__global__ void sanitize(
	unsigned char* __restrict__ strings,
	const unsigned int part,
	const unsigned int strings_lenght,
	const unsigned int search_size,
	const unsigned int results_offset,
	unsigned char* results)
{
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (offset < strings_lenght)
	{
		// strings: partA1, partA2, ... , partAN, partB1, partB2, ...
		//			------------ LINE ----------, -ITEM-, -ITEM-
		unsigned int os = offset * LINE_SIZE + part;
		unsigned int or = offset * TEXT_RESULT_VALUES_SIZE + results_offset;

		const unsigned char sanitize = ',';
		
		for (unsigned int i = 0; i < search_size; i++)
		{
			const unsigned char s = strings[os + i];
			
			results[or] = s == sanitize ? results[or] : s;
			or += s == sanitize ? 0 : 1;
		}
	}
}
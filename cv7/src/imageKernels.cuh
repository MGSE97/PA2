#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Converts color a float [pitch] linear memory - PREPARED FOR 2D CUDA BLOCK in generall,
/// 			but blockDim.y can be 1 as well.</summary>
/// <typeparam name="unsigned int srcBPP"> Source bits per pixel. </param>
/// <param name="src">   	Source data. </param>
/// <param name="srcWidth"> The width. </param>
/// <param name="srcHeight">The height. </param>
/// <param name="srcHeight">The pitch of src. </param>
/// <param name="dstPitch">	The pitch of dst. </param>
/// <param name="dst">   	[in,out] If non-null, destination for the. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int srcBPP>__global__ void colorToFloat(const unsigned char * __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst )
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((tx < srcWidth) && (ty < srcHeight))
	{
		unsigned int dstOffset = ty * dstPitch + tx;
		unsigned int srcOffset = (ty * srcPitch + tx) * srcBPP/8;
		unsigned int value = 0;

		if (srcBPP>=8)
		{
			value = src[srcOffset++];
		}
		if (srcBPP>=16)
		{
			value = (value<<8) | src[srcOffset++];
		}
		if (srcBPP>=24)
		{
			value = (value<<8) | src[srcOffset++];
		}
		if (srcBPP>=32)
		{
			value = (value<<8) | src[srcOffset++];
		}
		dst[dstOffset] = (float)value;
	}
}
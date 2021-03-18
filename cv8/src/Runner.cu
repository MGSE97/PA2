#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cudaDefs.h>
#include <glew.h>
#include <freeglut.h>
#include <imageManager.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <random>

#define BLOCK_DIM 32

#define TEXTURE_REFERENCE_API	0
#define TEXTURE_OBJECT_API	1
#define USED_API TEXTURE_REFERENCE_API	
#define USED_API TEXTURE_OBJECT_API	

// Configuration
unsigned int startPositionX = 4030;
unsigned int startPositionY = 3075 - 1450;
unsigned int next_time = 128;


std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0, 1);

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

//OpenGL
unsigned int viewportWidth = 1024;
unsigned int viewportHeight = 618;//1024;
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int imagePitch;
unsigned int pboID;
unsigned int textureID;
static int fpsCount = 0;
static int fpsLimit = 1;

StopWatchInterface* totalTimer = nullptr;
StopWatchInterface* workerTimer = nullptr;
char windowTitle[256];
float fps;
const int cbTimerDelay = 10;	//10 ms

//CUDA
cudaGraphicsResource_t cudaPBOResource;
cudaGraphicsResource_t cudaTexResource;

#if USED_API == TEXTURE_REFERENCE_API
	cudaChannelFormatDesc cudaTexChannelDesc;
	texture<uchar4, 2, cudaReadModeElementType> cudaTexRef;
#else
	cudaChannelFormatDesc cudaTexChannelDesc;
	cudaResourceDesc resDesc;
	cudaTextureDesc texDesc;
	cudaTextureObject_t cudaTex = 0;
#endif

unsigned int run_time = 0;

//OpenGL
void initGL(int argc, char** argv);
void releaseOpenGL();
void prepareTexture(const char* imageFileName);
void preparePBO();

//OpenGL Callback functions
void my_display();
void my_resize(GLsizei w, GLsizei h);
void my_idle();
void my_timer(int value);

//CUDA
void initCUDAtex();
void cudaWorker();
void releaseCUDA();

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);
	srand(time(NULL));

	sdkCreateTimer(&totalTimer);
	sdkResetTimer(&totalTimer);
	sdkCreateTimer(&workerTimer);
	sdkResetTimer(&workerTimer);

	initGL(argc, argv);
	//prepareTexture("D:/Documents/Projekty/Škola/PA2/cv8/assets/lena.png");
	prepareTexture("D:/Documents/Projekty/Škola/PA2/cv8/assets/world.png");
	preparePBO();

	initCUDAtex();

	//start rendering mainloop
	glutMainLoop();
	atexit([]()
		{
			sdkDeleteTimer(&totalTimer);
			sdkDeleteTimer(&workerTimer);
			releaseCUDA();
			releaseOpenGL();
		});
}

#pragma region OpenGL Routines

void initGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(viewportWidth, viewportHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Freeglut window");

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	//glutIdleFunc(my_idle);
	glutTimerFunc(cbTimerDelay, my_timer, 0);
	glutSetCursor(GLUT_CURSOR_CROSSHAIR);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);
	glViewport(0, 0, viewportWidth, viewportHeight);

	glFlush();
}

void prepareTexture(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);
	tmp = FreeImage_ConvertTo32Bits(tmp);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&textureID);
	glBindTexture( GL_TEXTURE_2D, textureID);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

void preparePBO()
{
	glGenBuffers(1, &pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);												// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, NULL,GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void releaseOpenGL()
{
	if (textureID > 0)
		glDeleteTextures(1, &textureID);
	if (pboID > 0)
		glDeleteBuffers(1, &pboID);
}

#pragma endregion

#pragma region OpenGL Callbacs

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glBegin(GL_QUADS);

	glTexCoord2d(0,0);		glVertex2d(0,0);
	glTexCoord2d(1,0);		glVertex2d(viewportWidth, 0);
	glTexCoord2d(1,1);		glVertex2d(viewportWidth, viewportHeight);
	glTexCoord2d(0,1);		glVertex2d(0, viewportHeight);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	glFlush();			
	glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
	viewportWidth=w; 
	viewportHeight=h; 

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0,0,viewportWidth,viewportHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0,viewportWidth, 0,viewportHeight);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void my_idle() 
{
	sdkStartTimer(&totalTimer);

	sdkStartTimer(&workerTimer);
	cudaWorker();
	sdkStopTimer(&workerTimer);

	glutPostRedisplay();
	sdkStopTimer(&totalTimer);

	if (++fpsCount == fpsLimit)
	{
		fps = 1000.0f / sdkGetAverageTimerValue(&totalTimer);
		sprintf(windowTitle, "Freeglut window (%d x %d): %.1f fps     worker: %.5f ms", viewportWidth, viewportHeight, fps, sdkGetAverageTimerValue(&workerTimer)/1000.f);
		glutSetWindowTitle(windowTitle);
		fpsCount = 0;
		fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
		sdkResetTimer(&totalTimer);
	}
}

void my_timer(int value)
{
	sdkStartTimer(&totalTimer);
	
	sdkStartTimer(&workerTimer);
	cudaWorker();
	sdkStopTimer(&workerTimer);
	
	glutPostRedisplay();
	sdkStopTimer(&totalTimer);

	if (++fpsCount == fpsLimit)
	{
		float fps = 1000.0f / sdkGetAverageTimerValue(&totalTimer);
		sprintf(windowTitle, "Freeglut window (%d x %d): %.1f fps     worker: %.5f ms", viewportWidth, viewportHeight, fps, sdkGetAverageTimerValue(&workerTimer) / 1000.f);

		glutSetWindowTitle(windowTitle);
		fpsCount = 0;
		fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
		sdkResetTimer(&totalTimer);
		sdkResetTimer(&workerTimer);
	}

	glutTimerFunc(cbTimerDelay, my_timer, 0);
}

#pragma endregion

#pragma region CUDA Routines

#if USED_API == TEXTURE_REFERENCE_API
__global__ void kernelRefAPI(const unsigned char time, const unsigned int pboWidth, const unsigned int pboHeight, unsigned char* pbo, unsigned int seedX, unsigned int seedY)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int pboElementOffset = (ty * pboWidth + tx) * 4;

	if ((tx < pboWidth) && (ty < pboHeight))
	{
		uchar4 texel = tex2D(cudaTexRef, tx, ty);
		bool valid = texel.x == 255 && texel.y == 255 && texel.z == 255;
		bool filled = texel.x == 255 && texel.y == 0 && texel.z == 0;
		if (valid && !filled)
		{
			bool fill = tx == seedX && ty == seedY;
			if (!fill)
			{
				int filled = 0;
				int empty = 0;
				for (unsigned int x = 0; x < 3; x++)
					for (unsigned int y = 0; y < 3; y++)
					{
						uchar4 next = tex2D(cudaTexRef, tx + x - 1, ty + y - 1);
						filled += next.x == 255 && next.y == 0 && next.z == 0;
						empty += next.x == 255 && next.y == 255 && next.z == 255;
					}
				fill = filled >= (time == 0 ? 1 : 3) || (filled > 0 && empty < 3);
			}
			texel.x = fill ? 255 : texel.x;
			texel.y = fill ? 0 : texel.y;
			texel.z = fill ? 0 : texel.z;
		}
		pbo[pboElementOffset++] = texel.x;
		pbo[pboElementOffset++] = texel.y;
		pbo[pboElementOffset++] = texel.z;
		pbo[pboElementOffset++] = texel.w;
	}
}
#endif


#if USED_API == TEXTURE_OBJECT_API
__global__ void kernelObjAPI(const unsigned char time, const unsigned int pboWidth, const unsigned int pboHeight, unsigned char* pbo, cudaTextureObject_t tex, unsigned int seedX, unsigned int seedY)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int pboElementOffset = (ty * pboWidth + tx) * 4;

	if ((tx < pboWidth) && (ty < pboHeight))
	{
		uchar4 texel = tex2D<uchar4>(tex, tx, ty);
		bool valid = texel.x == 255 && texel.y == 255 && texel.z == 255;
		bool filled = texel.x == 255 && texel.y == 0 && texel.z == 0;
		if (valid && !filled)
		{
			bool fill = tx == seedX && ty == seedY;
			if (!fill)
			{
				int filled = 0;
				int empty = 0;
				for (unsigned int x = 0; x < 3; x++)
					for (unsigned int y = 0; y < 3; y++)
					{
						uchar4 next = tex2D<uchar4>(tex, tx + x - 1, ty + y - 1);
						filled += next.x == 255 && next.y == 0 && next.z == 0;
						empty += next.x == 255 && next.y == 255 && next.z == 255;
					}
				fill = filled >= (time == 0 ? 1 : 3) || (filled > 0 && empty < 3);
			}
			texel.x = fill ? 255 : texel.x;
			texel.y = fill ? 0 : texel.y;
			texel.z = fill ? 0 : texel.z;
		}
		pbo[pboElementOffset++] = texel.x;
		pbo[pboElementOffset++] = texel.y;
		pbo[pboElementOffset++] = texel.z;
		pbo[pboElementOffset++] = texel.w;
	}
}
#endif

void initCUDAtex()
{
	cudaGLSetGLDevice(0);
	checkError();

	//Register main texture
	cudaGraphicsGLRegisterImage(&cudaTexResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);  // Map the GL texture resource with the CUDA resource
	checkError();

#if USED_API == TEXTURE_REFERENCE_API
	//cudaTexChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaTexRef.normalized = false;						//Otherwise TRUE to access with normalized texture coordinates
	cudaTexRef.filterMode = cudaFilterModePoint;		//Otherwise texRef.filterMode = cudaFilterModeLinear; for Linear interpolation of texels
	cudaTexRef.addressMode[0] = cudaAddressModeClamp;	//No repeat texture pattern
	cudaTexRef.addressMode[1] = cudaAddressModeClamp;	//No repeat texture pattern
#else
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;
#endif

	//Register PBO
	cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
	checkError();
}

void cudaWorker()
{
	cudaArray* array;
	cudaGraphicsMapResources(1, &cudaTexResource, 0);
	cudaGraphicsSubResourceGetMappedArray(&array, cudaTexResource, 0, 0);
	cudaGetChannelDesc(&cudaTexChannelDesc, array);

#if USED_API == TEXTURE_REFERENCE_API
	cudaBindTextureToArray(&cudaTexRef, array, &cudaTexChannelDesc);
	checkError();

	cudaGraphicsMapResources(1, &cudaPBOResource, 0);
	unsigned char* pboData;
	size_t pboSize;
	cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, cudaPBOResource);
	checkError();

	dim3 block = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	dim3 grid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	kernelRefAPI<<<grid, block>>>(run_time, imageWidth, imageHeight, pboData, startPositionX, startPositionY);

	cudaUnbindTexture(&cudaTexRef);

#else

	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = array;

	checkCudaErrors(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, NULL));

	checkError();

	cudaGraphicsMapResources(1, &cudaPBOResource, 0);
	unsigned char* pboData;
	size_t pboSize;
	cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, cudaPBOResource);
	checkError();

	dim3 block = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	dim3 grid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	kernelObjAPI<<<grid, block>>>(run_time, imageWidth, imageHeight, pboData, cudaTex, startPositionX, startPositionY);

	checkCudaErrors(cudaDestroyTextureObject(cudaTex));
#endif

	cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
	cudaGraphicsUnmapResources(1, &cudaTexResource, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory

	run_time++;
	if (run_time / next_time == 1)
	{
		run_time = 0;
		next_time = MAX(next_time * 0.8, 2);
		startPositionX = distribution(generator) * imageWidth;
		startPositionY = distribution(generator) * imageHeight;

		//printf(".");
	}
}

void releaseCUDA()
{
	cudaGraphicsUnregisterResource(cudaPBOResource);
	cudaGraphicsUnregisterResource(cudaTexResource);
}
#pragma endregion


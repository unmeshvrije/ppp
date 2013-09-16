
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstdio>

#include "common.h"

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

#define	OFFLOAD_KERNEL_HISTOGRAM	1
#define	OFFLOAD_KERNEL_RGB		1
#define	OFFLOAD_KERNEL_CONTRAST		1
#define	OFFLOAD_KERNEL_SMOOTH		1

#define	THREADS_PER_BLOCK	512
#define	THREADS_PER_BLOCK_X	32
#define	THREADS_PER_BLOCK_Y	16

#define	CONVOLUTION_TILE_SIZE	16// 32 gives worse results as 1024 threads block at barrier

#define	CONVOLUTION_TILE_SIZE_Y	32

#define	NEW_SMOOTH 1


#define	GET_GLOBAL_INDEX(INDEX, LR,UD, WIDTH, HEIGHT)			\
	(								\
		(							\
			((INDEX + (LR))/ WIDTH) != ((INDEX / WIDTH)) \
		)							\
		? -1:		\
		  (		\
			((INDEX + (LR) + ((WIDTH) * (UD))) < 0) ||		\
			((INDEX + (LR) + ((WIDTH) * (UD))) >= (WIDTH*HEIGHT)) \
		  )? -1 : (INDEX + LR + ((WIDTH) * (UD)))	\
	)




__global__ void
kernel_rgb2gray(
	unsigned char *inputImage,
	unsigned char *grayImage,
	const int width,
	const int height
	)
{
  int row;
  int col;

  row = blockIdx.y * blockDim.y + threadIdx.y;
  col = blockIdx.x * blockDim.x + threadIdx.x;

  int index = (row * width) + col;
  //int index = (col * width) + row;
  if (index >= (width * height))
  {
    return;
  }
  // Following condition is not valid when it comes to Image of size (not multiple of 32, 16)
  //if (row >= width || col >= height)
  /*
  Consider following situation
  Image 5 X 3
  every block has (can have) max 2 X 2 threads

  we want ceil ( 5 / 2 ) = 3
  	  ceil ( 3 / 2 ) = 2 i.e. (3 X 2 ) = 6 blocks

	 So that we have 24 threads (enough for 15 elements)

	 For block (1,0)
	 and thread(1,1)

	 row = 0 * 2 + 1 = 1
	 col = 1 * 2 + 1 = 3

	 now col >= height (3) so we would return if we had condition if (row >= width || col >= height)

	 but in fact,
	 index = row * width + col = 1 * 5 + 3 = 8 is valid (we have 15 elements)
  
  */

	// Kernel

  float grayPix = 0.0f;
  float r = static_cast< float >(inputImage[index]);
  float g = static_cast< float >(inputImage[(width * height) + index]);
  float b = static_cast< float >(inputImage[(2 * width * height) + index]);

  grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
  grayImage[index] = static_cast< unsigned char >(grayPix);
	// /Kernel
}

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, NSTimer &timer)
{
	cudaError_t devRetVal = cudaSuccess;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);

	#if OFFLOAD_KERNEL_RGB
	unsigned char *devInputImage;
	unsigned char *devGrayImage;

	int iImageSize = height * width * sizeof(unsigned char);

	devRetVal = cudaMalloc((void**)&devInputImage, iImageSize * 3);// for rgb
	if (cudaSuccess != devRetVal)
	{
	  cout << "Cannot allocate memory" << endl;
	  return;
	}
	devRetVal = cudaMalloc((void**)&devGrayImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
	  cout << "Cannot allocate memory" << endl;  
	  cudaFree(devInputImage);
	  return;
	}

	memoryTime.start();
	devRetVal = cudaMemcpy(devInputImage, inputImage, iImageSize * 3, cudaMemcpyHostToDevice);
	if (cudaSuccess != devRetVal)
	{
	  cout << "Cannot copy memory";
	  cudaFree(devInputImage);
	  cudaFree(devGrayImage);
	  return;
	}
	memoryTime.stop();

	// 32 * 16 = 512
	dim3 dimBlock(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y );

	int blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(THREADS_PER_BLOCK_X)));
	int blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(THREADS_PER_BLOCK_Y)));

	//cout << width << "/" << THREADS_PER_BLOCK_X << " = " << blockWidth << endl;
	//cout << height << "/" << THREADS_PER_BLOCK_Y << " = " << blockHeight << endl;

	dim3 dimGrid(blockWidth,blockHeight);// width * height number of blocks
	
	kernelTime.start();
	kernel_rgb2gray<<<dimGrid, dimBlock>>>(devInputImage, devGrayImage, width, height);
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess )
	{
	  cerr << "Uh, the kernel had some kind of issue: " << devRetVal << endl;
	  cudaFree(devInputImage);
	  cudaFree(devGrayImage);
	  return;
	}

	
	memoryTime.start();
	devRetVal = cudaMemcpy(grayImage, devGrayImage, iImageSize, cudaMemcpyDeviceToHost);
	if (cudaSuccess != devRetVal)
	{
	  cout << "Cannot copy memory";
	  cudaFree(devInputImage);
	  cudaFree(devGrayImage);

	  return;
	}
	memoryTime.stop();

	cudaFree(devInputImage);
	cudaFree(devGrayImage);

	#else
	kernelTime.start();
	for (int y = 0; y < height; ++y)
	{
	  for (int x = 0; x < width; ++x)
	  {
  	   float grayPix = 0.0f;
	   float r = static_cast< float >(inputImage[(y * width) + x]);
	   float g = static_cast< float >(inputImage[(y * height) + (y * width) + x]);
	   float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

	   grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
	   grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
	  }
	}
	kernelTime.stop();
	#endif

	cout << fixed << setprecision(6);
	cout << "rgb2gray (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "rgb2gray (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;
}
/*
__global__ void
kernel_histogram_shm(
		unsigned char *grayImage,
		unsigned int *histogram,
		int width,
		int height
		)
{
  // All threads in a block will use this
  __shared__ unsigned int shHistogram[HISTOGRAM_SIZE];
  int threadIndexWithinBlock = threadIdx.x;
  shHistogram[threadIndexWithinBlock] = 0;
  __syncthreads();

 // if (row >= width || col >= height)
 //   return;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x + gridDim.x;
  while (index < (width * height))
  {
    atomicAdd(&shHistogram[static_cast<int>(grayImage[index])], 1);
    index += stride;
  }
  __syncthreads();

  atomicAdd(&histogram[threadIndexWithinBlock], shHistogram[threadIndexWithinBlock]);
  //atomicAdd(&shHistogram[static_cast< unsigned int >(grayImage[index])], 1);
 // __syncthreads();

  //vectorAdd(shHistogram, histogram);
}
*/
__global__ void
kernel_histogram(
		unsigned char *grayImage,
		unsigned int *histogram,
		int width,
		int height
		)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= width || col >= height)
    return;

  int index = row * width + col;

  atomicAdd(&histogram[static_cast<int>(grayImage[index])], 1);
}

/*
  Reduction kernel using shared memory for histogram calculation
*/
__global__ void
kernel_histogram_shm(
	unsigned char *grayImage,
	unsigned int *histogram,
	int width,
	int height
	)
{
//  int row = blockIdx.y * blockDim.y + threadIdx.y;
//  int col = blockIdx.x * blockDim.x + threadIdx.x;
  //int thread

//  if (row >= width || col >= height)
  //  return;

  __shared__ unsigned int shHistogram[HISTOGRAM_SIZE];
  // there are 512 threads per block 

  int threadIndexWithinBlock = threadIdx.x;

  if (blockDim.x > 256)
  {
    threadIndexWithinBlock = threadIdx.x / 2;
  }

  
  shHistogram[threadIndexWithinBlock] = 0;
  __syncthreads();

  int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  //if (threadIndex == 0)
  //printf("Stride = %d\n", stride);

  if (threadIndex >= (width * height))
    return;

  /*
  Ensure coalesced access to global memory with the use of strides
  */
  while (threadIndex < (width * height))
  {
    atomicAdd(&shHistogram[static_cast<int>(grayImage[threadIndex])],1);
    threadIndex += stride;
  }

  __syncthreads();

  atomicAdd(&histogram[threadIndexWithinBlock], shHistogram[threadIndexWithinBlock]);
}

void
histogram1D(
	unsigned char *grayImage,
	unsigned char *histogramImage,
	const int width,
	const int height,
	unsigned int *histogram,
	const unsigned int HISTOGRAM_SIZE,
	const unsigned int BAR_WIDTH,
	NSTimer &timer,
	int SpecialStride
	)
{
	cudaError_t devRetVal = cudaSuccess;
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);
	
	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));


	#if OFFLOAD_KERNEL_HISTOGRAM
	int iImageSize = height * width * sizeof(unsigned char);

	unsigned char *devGrayImage;
	unsigned int *devHistogram;
	devRetVal = cudaMalloc((void**) &devGrayImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot allocate memory" << endl;
	  return;
	}

	devRetVal = cudaMalloc((void**) &devHistogram, HISTOGRAM_SIZE * sizeof(int));
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot allocate memory" << endl;
	  cudaFree(devGrayImage);
	  return;
	}

	memoryTime.start();
	devRetVal = cudaMemcpy(devGrayImage, grayImage, iImageSize, cudaMemcpyHostToDevice);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devHistogram);
	  return;
	}

	devRetVal = cudaMemcpy(devHistogram, histogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devHistogram);
	  return;
	}
	memoryTime.stop();

	int iThreadsPerBlock = 256;
	//512 does not give better speedups
	//More waiting due to __synchronize()


        dim3 dimBlock(iThreadsPerBlock);

	int blockWidth = static_cast<int>(ceil(height * width / static_cast<float>(iThreadsPerBlock)));

	//
	// For image09.bmp, SpecialStride has to be at least 8 or above
	// Any dimension of grid should not exceed 65535
	//
	blockWidth /= SpecialStride;
	int blockHeight = 1;
	dim3 dimGrid(blockWidth, blockHeight);// width * height number of blocks

	kernelTime.start();
	//kernel_histogram<<<dimGrid, dimBlock>>>(devGrayImage, devHistogram, width, height);
	kernel_histogram_shm<<<dimGrid, dimBlock>>>(devGrayImage, devHistogram, width, height);

	//
	// Can be launched with #of SMs * 2 as grid size
	//
	//kernel_histogram_shm<<<30, dimBlock>>>(devGrayImage, devHistogram, width, height);
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess )
	{
	  cerr << "Uh, the kernel had some kind of issue." << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devHistogram);
	  return;
	}

	memoryTime.start();
	cudaMemcpy(histogram, devHistogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devHistogram);
	  return;
	}

	memoryTime.stop();

	cudaFree(devGrayImage);
	cudaFree(devHistogram);
	#else

	// Kernel
	kernelTime.start();
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			histogram[static_cast< unsigned int >(grayImage[(y * width) + x])] += 1;
		}
	}
	kernelTime.stop();
	// EO/Kernel
	#endif

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) {
		if ( histogram[i] > max ) {
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) {
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) {
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) {
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}
	
	cout << fixed << setprecision(6);
	cout << "histogram1D (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "histogram1D (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;
}

__global__ void
kernel_contrast(
	unsigned char *grayImage,
	float min,
	float max,
	float diff,
	int width,
	int height
	)
{
  int row;
  int col;

  row = blockIdx.y * blockDim.y + threadIdx.y;
  col = blockIdx.x * blockDim.x + threadIdx.x;


  int index = (row * width) + col;

  if (row >= height || col >=width)
  {
    return;
  }

  unsigned char pixel = grayImage[index];

  if ( pixel < min )
  {
	pixel = 0;
  }
  else if ( pixel > max )
  {
    pixel = 255;
  }
  else
  {
    pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
  }				

  grayImage[index] = pixel;
}


void 
contrast1D(
	unsigned char *grayImage,
	const int width,
	const int height,
	unsigned int *histogram,
	const unsigned int HISTOGRAM_SIZE,
	const unsigned int CONTRAST_THRESHOLD,
	NSTimer &timer
	)
{
	cudaError_t devRetVal = cudaSuccess;
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	#if OFFLOAD_KERNEL_CONTRAST

	int iImageSize = width * height *sizeof(unsigned char);
	unsigned char *devGrayImage;

	devRetVal = cudaMalloc((void**)&devGrayImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot allocate memory" << endl;
	  return;
	}

	memoryTime.start();
	devRetVal = cudaMemcpy(devGrayImage, grayImage, iImageSize,cudaMemcpyHostToDevice);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy  memory" << endl;
	  cudaFree(devGrayImage);
	  return;
	}
	memoryTime.stop();

        dim3 dimBlock(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y );
	int blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(THREADS_PER_BLOCK_X)));
	int blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(THREADS_PER_BLOCK_Y)));
	dim3 dimGrid(blockWidth,blockHeight);// width * height number of blocks

	kernelTime.start();
	kernel_contrast<<<dimGrid, dimBlock>>>(devGrayImage, min, max, diff, width, height);
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess )
	{
	  cerr << "Uh, the kernel had some kind of issue." << endl;
	  cudaFree(devGrayImage);
	  return;
	}

	memoryTime.start();
	devRetVal = cudaMemcpy(grayImage, devGrayImage, iImageSize, cudaMemcpyDeviceToHost);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  return;
	}
	memoryTime.stop();

	cudaFree(devGrayImage);

	#else
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for (int x = 0; x < width; x++ ) {
			unsigned char pixel = grayImage[(y * width) + x];

			if ( pixel < min ) {
				pixel = 0;
			}
			else if ( pixel > max ) {
				pixel = 255;
			}
			else {
				pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
			}
			
			grayImage[(y * width) + x] = pixel;
		}
	}
	// /Kernel
	kernelTime.stop();
	#endif
	
	cout << fixed << setprecision(6);
	cout << "contrast1D (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "contrast1D (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;
}


__constant__ float constFilter[FILTER_WIDTH * FILTER_WIDTH];

__global__ void
kernel_smooth(
	unsigned char *grayImage,
	unsigned char *smoothImage,
	int filterSize,
	int width,
	int height,
	int oldWidth
	)
{

  int row;
  int col;
  
  row = blockIdx.y * blockDim.y + threadIdx.y;
  col = blockIdx.x * blockDim.x + threadIdx.x;

 int index = row * width + col;
 
 if (row >= height || col >= width)
 {
   return;
 }

  unsigned int filterItem = 0;
  float filterSum = 0.0f;
  float smoothPix = 0.0f;

  for ( int fy = row - 2; fy < row + 3; fy++ )
  {
    for ( int fx = col - 2; fx < col + 3; fx++ )
    {
      if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= oldWidth)) )// fx >= width
      {
          filterItem++;
	  continue;
      }
      
      /*if ( (fy < 0) || (fx < 0) || ((fy*width + fx) >= (width * height)) )
      {
          filterItem++;
	  continue;
      }*/

      smoothPix += grayImage[(fy * width) + fx] * constFilter[filterItem];
      filterSum += constFilter[filterItem];
      filterItem++;
    }
  }

  smoothPix /= filterSum;
  smoothImage[index] = static_cast< unsigned char >(smoothPix);
}

#if NEW_SMOOTH
template<short int tileSize>
__global__ void
kernel_smooth_convolution(
	unsigned char *grayImage,
	unsigned char *smoothImage,
	int width,
	int height,
	int oldWidth
	)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int index = row * width + col;

  //
  // All threads in a block form a TILE
  // they all load their respective elements and
  // GHOST or HALO elements
  // When all the elements are loaded (in shared memory)
  // all threads proceed with the simple loop
  //
  // BlockDim would be 16 * 16  or 32 *32 mostly

  short int tileDimx = blockDim.x  + FILTER_WIDTH - 1;
  short int tileDimy = blockDim.y + FILTER_WIDTH - 1;

  short int shIndex;
  int globalIndex;

  //if (index >= (width * height))
  //  return;

  //
  // Shared memory Sub-Matrix of Gray Image
  // Will be sent through template parameter tileSize
  //
  __shared__ unsigned char shGrayImageTile[tileSize];

  __shared__ unsigned char shValidElement[tileSize];//TODO

  short int n = FILTER_WIDTH / 2;

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
   for (int i = 0; i < (tileDimx * tileDimx); ++i)
     shValidElement[i] = 0;
  }

  __syncthreads();

  //
  // Do not forget to load OWN element of thread
  //
  shIndex =( (threadIdx.y+2) *tileDimy ) + (threadIdx.x + 2);

   if (index >= (width * height))
     shGrayImageTile[shIndex] = 0;
   else
   {
     shGrayImageTile[shIndex] = grayImage[index];

     //TODO: Decision whether this element is from PADDED column or proper column
     if ( col < oldWidth  ) 	  
     {
     // it is valid element, we should note it, because
     // we need to identify this in for loop of filtering
     //
     // Note that we need to add filter[filtersum] to filterSum for every valid element
     // filterSum += filter[filterItem];
     //
      shValidElement[shIndex] = 1;
     }
   }

  /*
  
  X # | X #   Q'
  # # | # #   P'
  --- +-----------------
  X # | A B...Q
  # # | C D...P

  Thread A (which is at corner of a tile)
  will load 3 elements from global memory
  each of which is 2 rows/columns/diagonal units away from him

  Same is true of other threads in this part of tile

  Threads P and Q will load one element each (P' and Q' respectively)

  */

  // For computing diagonal ghost elements
  //+
  if
  (
  	( 0 == threadIdx.x || 1 == threadIdx.x)&&
  	( 0 == threadIdx.y || 1 == threadIdx.y)
  )
  {
    // Left top block of 2X2 threads
    //Diagonal
    shIndex = threadIdx.y * tileDimy + threadIdx.x;
    globalIndex = GET_GLOBAL_INDEX(index,-n,-n, width, height);// LR, UD, width, height

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }
  else  if
  (
  	( 0 == threadIdx.x || (1 == threadIdx.x))&&
  	( blockDim.y-2 == threadIdx.y || (blockDim.y-1 == threadIdx.y))
  )
  {
    // thread (1,6) will actually compute 6th row , 1st column
    // which is left bottom
    shIndex = ((threadIdx.y + 4) * tileDimy) + (threadIdx.x );
    globalIndex = GET_GLOBAL_INDEX(index,-n,n, width, height);// LR, UD, width, height
    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }
  else  if
  (
  	( blockDim.x-2 == threadIdx.x || (blockDim.x-1 == threadIdx.x))&&
  	( 0 == threadIdx.y || (1 == threadIdx.y))
  )
  {
    // right top
    shIndex = threadIdx.y * tileDimy + (threadIdx.x + 4);
    globalIndex = GET_GLOBAL_INDEX(index,n,-n, width, height);// LR, UD, width, height

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }
  else  if
  (
  	( blockDim.x-2 == threadIdx.x || (blockDim.x-1 == threadIdx.x))&&
  	( blockDim.y-2 == threadIdx.y || (blockDim.y-1 == threadIdx.y))
  )
  {
    // right bottom
    shIndex = ((threadIdx.y + 4) * tileDimy) + (threadIdx.x + 4);
    globalIndex = GET_GLOBAL_INDEX(index,n,n, width, height);// LR, UD, width, height

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }

  //-
  // for computing diagonal ghost elements


  // Top left, bottom left,... also participate in usual right/left/top/bottom ghost element loading
  if (0 == threadIdx.y || 1 == threadIdx.y)
  {
    // UP
    shIndex = (threadIdx.y * tileDimy) + threadIdx.x  +2;
    globalIndex = GET_GLOBAL_INDEX(index,0,-n, width, height);

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }
  else if(blockDim.y-2 == threadIdx.y || blockDim.y-1 == threadIdx.y)
  {
    //down
    shIndex = ( (threadIdx.y + 4) * tileDimy) + threadIdx.x  +2;
    globalIndex = GET_GLOBAL_INDEX(index,0,n, width, height);

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }
  }

   if (0 == threadIdx.x || 1 == threadIdx.x)
  {
    // Left
    shIndex = (threadIdx.y + n) * tileDimy + threadIdx.x;
    globalIndex = GET_GLOBAL_INDEX(index,-n,0, width, height);

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

  }
  else if(blockDim.x-2 == threadIdx.x || blockDim.x-1 == threadIdx.x)
  {
   //right
    shIndex = ((threadIdx.y + 2) * tileDimy) + threadIdx.x  + 4;
    globalIndex = GET_GLOBAL_INDEX(index,n,0, width, height);

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

  }


  //
  // Wait till all threads in the block, load the elements they are responsible for.
  //
  __syncthreads();

  unsigned int filterItem = 0;
  float filterSum = 0.0f;
  float smoothPix = 0.0f;

  for ( int i = 0; i < FILTER_WIDTH; i++ )
  {
    for ( int j =0; j < FILTER_WIDTH; j++ )
    {

       if (shValidElement[(threadIdx.y + i) * tileDimy + ( threadIdx.x + j)] == 0)//invalid element
      {
	//
	// 0 in Gray Image Tile has three meanings:
	// 1. Valid Gray Image pixel
	// 2. PADded element (invalid)
	// 3. GHOST element  (invalid)
	//
      if (shGrayImageTile[(threadIdx.y + i) * tileDimy + ( threadIdx.x + j)] == 0)//invalid element
       {
          filterItem++;
	  continue;
       }
      }
      
  	//if (index == 7039999)
  	//if (index == 0)
	{
	  //printf("Tile[%d][%d] = %d\n",(threadIdx.y + i)*tileDimy, (threadIdx.x + j), shGrayImageTile[((threadIdx.y + i) * tileDimy) + (threadIdx.x + j)]);
	}
      smoothPix += shGrayImageTile[((threadIdx.y + i) * tileDimy) + (threadIdx.x + j)] * constFilter[filterItem];
      
      filterSum += constFilter[filterItem];
      filterItem++;
    }
  }

  smoothPix /= filterSum;
  if (index < (width * height))
  smoothImage[index] = static_cast< unsigned char >(smoothPix);
}
#endif

template<short int tileSize>
__global__ void
kernel_smooth_perfect(
	unsigned char *grayImage,
	unsigned char *smoothImage,
	int width,
	int height,
	int oldWidth
	)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int index = row * width + col;

  //
  // All threads in a block form a TILE
  // they all load their respective elements and
  // GHOST or HALO elements
  // When all the elements are loaded (in shared memory)
  // all threads proceed with the simple loop
  //
  // BlockDim would be 16 * 16  or 32 *32 mostly

  //short int tileDimx = blockDim.x + FILTER_WIDTH - 1;
  //short int tileDimy = blockDim.y + FILTER_WIDTH - 1;

  short int shIndex;
  int globalIndex;

  
//  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x ==0 && blockIdx.y == 0)
  //  printf("blockDim(%d,%d) ,tile = %d \n", blockDim.x, blockDim.y, tileSize);
  //if (index >= (width * height))
  //  return;

  //
  // Shared memory Sub-Matrix of Gray Image
  // Will be sent through template parameter tileSize
  //
  __shared__ unsigned char shGrayImageTile[tileSize*tileSize];
  __shared__ unsigned char shValidElement[tileSize *tileSize];//TODO

  short int n = FILTER_WIDTH / 2;

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
   for (int i = 0; i < (tileSize * tileSize); ++i)
     shValidElement[i] = 0;
  }

  __syncthreads();

  /*
  
  X # | # # X
  # # | # # #
  --- +-----------------
  # # | A # #
  # # | # # #
  X # | # # X	  | 

  Each Thread A will load 4 elements from tile
  each 2 units diagonally away from it
  Same is true of other threads in this part of tile

  Threads P and Q will load one element each (P' and Q' respectively)

  */

    // Left top
    shIndex = threadIdx.y * tileSize + threadIdx.x;
    globalIndex = GET_GLOBAL_INDEX(index,-n,-n, width, height);// LR, UD, width, height

    if (globalIndex < 0 || col-n < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
        if (col-n < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

    //	Left bottom
    shIndex = ((threadIdx.y + 4) * tileSize) + (threadIdx.x );
    globalIndex = GET_GLOBAL_INDEX(index,-n,n, width, height);// LR, UD, width, height
    if (globalIndex < 0 || col-n < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];

      if (col-n < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

     // Right top
    shIndex = threadIdx.y * tileSize + (threadIdx.x + 4);
    globalIndex = GET_GLOBAL_INDEX(index,n,-n, width, height);// LR, UD, width, height

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      if (col+n < oldWidth) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

    // Right bottom
    shIndex = ((threadIdx.y + 4) * tileSize) + (threadIdx.x + 4);
    globalIndex = GET_GLOBAL_INDEX(index,n,n, width, height);// LR, UD, width, height

    if (globalIndex < 0)
    {
      shGrayImageTile[shIndex] = 0;
    }
    else
    {
      shGrayImageTile[shIndex] = grayImage[globalIndex];
      if (col+n < oldWidth ) // Means it is not padded element
        shValidElement[shIndex] =1;
    }

  //
  // Wait till, all threads in the block, load the elements they are responsible for.
  //
  __syncthreads();

  unsigned int filterItem = 0;
  float filterSum = 0.0f;
  float smoothPix = 0.0f;

  for ( int i = 0; i < FILTER_WIDTH; i++ )
  {
    for ( int j =0; j < FILTER_WIDTH; j++ )
    {

       if (shValidElement[(threadIdx.y + i) * tileSize + ( threadIdx.x + j)] == 0)//invalid element
      {
	//
	// 0 in Gray Image Tile has three meanings:
	// 1. Valid Gray Image pixel
	// 2. PADded element (invalid)
	// 3. GHOST element  (invalid)
	//
      if (shGrayImageTile[(threadIdx.y + i) * tileSize + ( threadIdx.x + j)] == 0)//invalid element
       {
          filterItem++;
	  continue;
       }
      }
  
      smoothPix += shGrayImageTile[((threadIdx.y + i) * tileSize) + (threadIdx.x + j)] * constFilter[filterItem];
      
      filterSum += constFilter[filterItem];
      filterItem++;
    }
  }

  smoothPix /= filterSum;
  if (index < (width * height))
  smoothImage[index] = static_cast< unsigned char >(smoothPix);
}


/*
  This kernel is not working! Version(2) I guess
  It tries to read image in 32-bit access pattern.
  It *can* lead to good results if correctly implemented!
*/
template<short int tileSize>
__global__ void
kernel_smooth_tweak(
	unsigned char *grayImage,
	unsigned char *smoothImage,
	int width,
	int height,
	int oldWidth
	)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  col = col *4;
  int index = row * width + col;

  //
  // All threads in a block form a TILE
  // they all load their respective elements and
  // GHOST or HALO elements
  // When all the elements are loaded (in shared memory)
  // all threads proceed with the simple loop
  //
  // BlockDim would be 16 * 16  or 32 *32 mostly

  short int tileWidth = (blockDim.x  * 4) + 8; // FILTER_WIDTH - 1; // 72
  short int tileHeight = blockDim.y + FILTER_WIDTH - 1; // 20

  int shIndex = -1;
  int globalIndex = -1;

  //if (index >= (width * height))
  //  return;
  if (threadIdx.x == 1 && threadIdx.y == 2 && blockIdx.x ==0 && blockIdx.y == 0)
  {
    printf("index = %d\n", index);
    printf("tileWid = %d\n", tileWidth);
    printf("tileHeight = %d\n", tileHeight);
    printf("tileSize = %d\n", tileSize);
  }
  //
  // Shared memory Sub-Matrix of Gray Image
  // Will be sent through template parameter tileSize
  //
  __shared__ unsigned char shGrayImageTile[tileSize]; // tileSize = (72 * 20) assume

  __shared__ unsigned char shValidElement[tileSize];//TODO

  short int n = FILTER_WIDTH / 2;

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
   for (int i = 0; i < (tileWidth * tileHeight); ++i)
     shValidElement[i] = 0;
  }

  __syncthreads();

  //
  // Do not forget to load OWN element of thread
  //
  shIndex =( (threadIdx.y+2) *tileWidth ) + ((threadIdx.x + 1)*4);

 // if (threadIdx.x == 1 && threadIdx.y == 2 && blockIdx.x ==0 && blockIdx.y == 0)
  {
//	printf("shIndex = %d\n", shIndex);
  }

   if (index >= (width * height))
     shGrayImageTile[shIndex] = 0;
   else
   {
     *((unsigned int*) &shGrayImageTile[shIndex]) = *((unsigned int *)(&grayImage[index]));

     //TODO: Decision whether this element is from PADDED column or proper column
     if ( col < oldWidth  ) 	  
     {
     // it is valid element, we should note it, because
     // we need to identify this in for loop of filtering
     //
     // Note that we need to add filter[filtersum] to filterSum for every valid element
     // filterSum += filter[filterItem];
     //
      shValidElement[shIndex] = 1;
     }

     if (col+1 < oldWidth)
       shValidElement[shIndex +1] = 1;
     if (col+2 < oldWidth)  
       shValidElement[shIndex +2] = 1;
     if (col+3 < oldWidth)  
       shValidElement[shIndex +3] = 1;
   }

  /*
  
  X # | X #   Q'
  # # | # #   P'
  --- +-----------------
  X # | A B...Q
  # # | C D...P

  Thread A (which is at corner of a tile)
  will load 3 elements from global memory
  each of which is 2 rows/columns/diagonal units away from him

  Same is true of other threads in this part of tile

  Threads P and Q will load one element each (P' and Q' respectively)

  */

  // For computing diagonal ghost elements
  //+
  if
  (
  	( 0 == threadIdx.x)&&
  	( 0 == threadIdx.y || 1 == threadIdx.y)
  )
  {
    // Left top block of 2X2 threads
    //Diagonal
    shIndex = threadIdx.y * tileWidth + (threadIdx.x * 4);
    globalIndex = GET_GLOBAL_INDEX(index,-4,-n, width, height);// LR, UD, width, height

    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }

    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *) &shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }
  else  if
  (
  	( 0 == threadIdx.x)&&
  	( blockDim.y-2 == threadIdx.y || (blockDim.y-1 == threadIdx.y))
  )
  {
    // thread (1,6) will actually compute 6th row , 1st column
    // which is left bottom
    shIndex = ((threadIdx.y + 4) * tileWidth) + (threadIdx.x * 4);
    globalIndex = GET_GLOBAL_INDEX(index,-4,n, width, height);// LR, UD, width, height
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *) &shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }
  else  if
  (
  	(  (blockDim.x-1 == threadIdx.x))&&
  	( 0 == threadIdx.y || (1 == threadIdx.y))
  )
  {
    // right top
    shIndex = threadIdx.y * tileWidth + ((threadIdx.x +2)* 4);
    globalIndex = GET_GLOBAL_INDEX(index,4,-n, width, height);// LR, UD, width, height
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }
  else  if
  (
  	( (blockDim.x-1 == threadIdx.x))&&
  	( blockDim.y-2 == threadIdx.y || (blockDim.y-1 == threadIdx.y))
  )
  {
    // right bottom
    shIndex = ((threadIdx.y + 4) * tileWidth) + ((threadIdx.x+2)*  4);
    globalIndex = GET_GLOBAL_INDEX(index,4,n, width, height);// LR, UD, width, height
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
  //    if (globalIndex < (width*height)-3 && shIndex < ((72*20)-3)) 
      *((unsigned int *) (&shGrayImageTile[shIndex])) = *(unsigned int*)(&grayImage[globalIndex]);
    //  else
     //  printf("gb index= %d, thread (%d,%d) , block(%d,%d)\n", globalIndex,threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }

  }

  //-
  // for computing diagonal ghost elements


  // Top left, bottom left,... also participate in usual right/left/top/bottom ghost element loading
  if (0 == threadIdx.y || 1 == threadIdx.y)
  {
    // UP
    shIndex = (threadIdx.y * tileWidth) + (threadIdx.x+1)*4;
    globalIndex = GET_GLOBAL_INDEX(index,0,-n, width, height);
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }

  }
  else if(blockDim.y-2 == threadIdx.y || blockDim.y-1 == threadIdx.y)
  {
    //down
    shIndex = ( (threadIdx.y + 4) * tileWidth) + (threadIdx.x + 1) *4;
    globalIndex = GET_GLOBAL_INDEX(index,0,n, width, height);
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }

   if (0 == threadIdx.x)
  {
    // Left
    shIndex = (threadIdx.y + n) * tileWidth + threadIdx.x*4;
    globalIndex = GET_GLOBAL_INDEX(index,-4,0, width, height);
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }
  else if(blockDim.x-1 == threadIdx.x)
  {
   //right
    shIndex = ((threadIdx.y + n) * tileWidth) + (threadIdx.x+2) * 4;
    globalIndex = GET_GLOBAL_INDEX(index,4,0, width, height);
    if (shIndex >= (72 * 20))
    {
      printf("shIndex = %d, thread (%d,%d)\n", shIndex, threadIdx.x, threadIdx.y);
    }
    if (globalIndex < 0)
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = 0;//32 bits write
    }
    else
    {
      *((unsigned int *)&shGrayImageTile[shIndex]) = *(unsigned int*)(&grayImage[globalIndex]);
      // If global index is -1, then grayImage[-1] is sometimes turned out to be '0' No seg fault :(. Hence the check
      for (int i = 0; i < 4; ++i)
      {
        if (col+i < oldWidth) // Means it is not padded element
          shValidElement[shIndex+i] =1;
      }
    }
  }


  //
  // Wait till all threads in the block, load the elements they are responsible for.
  //
  __syncthreads();

  //
  // Now here we will find Smooth pixel for 4 pixels (32 bits)
  // as each thread is responsible for 4 pixels now
  //

  unsigned int result = 0;
  unsigned int shift = 0x000000ff;

  if (index  == 1048572 || index == 4)
  {
    printf("index = %d=> thread(%d,%d), block(%d, %d)\n", index, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
  }

  for (int k = 2; k <= 5; ++k)
  {
   unsigned int filterItem = 0;
   float filterSum = 0.0f;
   float smoothPix = 0.0f;
   
   for ( int i = 0; i < FILTER_WIDTH; i++ )
   {
    for ( int j =0; j < FILTER_WIDTH; j++ )
    {

       if (shValidElement[(threadIdx.y + i) * tileWidth + ( (threadIdx.x * 4 + k) + j)] == 0)//invalid element
      {
	//
	// 0 in Gray Image Tile has three meanings:
	// 1. Valid Gray Image pixel
	// 2. PADded element (invalid)
	// 3. GHOST element  (invalid)
	//
      if (shGrayImageTile[(threadIdx.y + i) * tileWidth + ( (threadIdx.x * 4 +k) + j)] == 0)//invalid element
       {
          filterItem++;
	  continue;
       }
      }
      
      if(
         (index == 0 && k ==2)||
         (index == 1048572 && k ==2)
	 )
      {
        printf("row = %d, col = %d\n", (threadIdx.y+i)*tileWidth,  (threadIdx.x * 4 )+ k+j);
      }
      smoothPix += shGrayImageTile[((threadIdx.y + i) * tileWidth) + ((threadIdx.x * 4 + k) + j)] * constFilter[filterItem];
      
      filterSum += constFilter[filterItem];
      filterItem++;
    }
   }

   smoothPix /= filterSum;
   unsigned char temp = static_cast<unsigned char>(smoothPix);

   result |= (shift & temp);
   shift <<= 8;
  }

  if (index < (width * height))
  *((unsigned int*)(&smoothImage[index])) = result;
}

void
triangularSmooth(
	unsigned char *grayImage,
	unsigned char *smoothImage,
	const int width,
	const int height,
	const float *filter,
	const int filterSize,
	NSTimer &timer,
	int kernelToLaunch
	)
{
	cudaError_t devRetVal;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);	
	NSTimer memoryTime = NSTimer("memoryTime", false, false);	


	#if OFFLOAD_KERNEL_SMOOTH

	int iImageSize = width * height * sizeof(unsigned char);


	unsigned char *devGrayImage;
	unsigned char *devSmoothImage;

	//+ Padding
	bool padding = false;
	int newWidth = 0;
	unsigned char *newGrayImage = NULL;
	unsigned char *newSmoothImage = NULL;

	if (width % CONVOLUTION_TILE_SIZE != 0)
	{
	  // Padding required
	  padding = true;

	  newWidth = PadImage(grayImage, width, height, CONVOLUTION_TILE_SIZE, &newGrayImage);
	 // cout << "New width = " << newWidth << endl;
	  iImageSize = newWidth * height * sizeof(unsigned char);
	  newSmoothImage = new unsigned char[iImageSize];
	}
	else
	{
	  newWidth = width;
	}

	//-

	devRetVal = cudaMalloc((void**)&devGrayImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot allocate memory" << endl;
	  return;
	}

	devRetVal = cudaMalloc((void**)&devSmoothImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot allocate memory" << endl;
	  cudaFree(devGrayImage);
	  return;
	}

	memoryTime.start();
	if (true == padding)
	{
	  devRetVal = cudaMemcpy(devGrayImage, newGrayImage, iImageSize, cudaMemcpyHostToDevice);
	}
	else
	{
	  devRetVal = cudaMemcpy(devGrayImage, grayImage, iImageSize, cudaMemcpyHostToDevice);
	}

	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devSmoothImage);

	  return;
	}

	devRetVal = cudaMemcpyToSymbol(constFilter, filter, filterSize * sizeof(float));
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory to constant memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devSmoothImage);
	  return;
	}

	memoryTime.stop();

	dim3 dimBlock;
	int blockWidth;
	int blockHeight;
	dim3 dimGrid;

	if (0 == kernelToLaunch)
	{
	  cout << "Launching kernel with constant memory optimization" << endl;
	  // TODO:
	  dimBlock = dim3(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y );
	  blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(THREADS_PER_BLOCK_X)));
	  blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(THREADS_PER_BLOCK_Y)));
	  dimGrid = dim3(blockWidth,blockHeight);// width * height number of blocks
	}
	else if (2 == kernelToLaunch)
	{
	  // tweak kernel here
	  // Not working currently
	  cout << "Launching 32 bit access smoothing" << endl;
	  dimBlock = dim3(CONVOLUTION_TILE_SIZE, CONVOLUTION_TILE_SIZE); // 16 * 16
	  // TODO: calculate properly
	  //blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(CONVOLUTION_TILE_SIZE * 2)));
	  //blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(CONVOLUTION_TILE_SIZE * 2)));
	  blockWidth = 64;
	  blockHeight = 16;
	  cout << "Block width " << blockWidth << "block height" << blockHeight << endl;
	  dimGrid = dim3(16,64);// width * height number of blocks
	}
	else if (3 == kernelToLaunch)
	{
	  cout << "Launching shared memory convolution kernel with BETTER access" << endl;
	  dimBlock = dim3(CONVOLUTION_TILE_SIZE, CONVOLUTION_TILE_SIZE); // 16 * 16
	  blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(CONVOLUTION_TILE_SIZE )));
	  
	  blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(CONVOLUTION_TILE_SIZE )));
	  dimGrid = dim3(blockWidth,blockHeight);// width * height number of blocks
	}
	else
	{
	   // for kernel 1
	  cout << "Launching shared memory convolution kernel" << endl;
	  dimBlock = dim3(CONVOLUTION_TILE_SIZE, CONVOLUTION_TILE_SIZE); // 16 * 16
	  blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(CONVOLUTION_TILE_SIZE )));
	  
	  blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(CONVOLUTION_TILE_SIZE )));
	  dimGrid = dim3(blockWidth,blockHeight);// width * height number of blocks
	}

	const short int tileDim = CONVOLUTION_TILE_SIZE + FILTER_WIDTH - 1;

	if (0 == kernelToLaunch)
	{
	  kernelTime.start();
	  kernel_smooth<<<dimGrid, dimBlock>>>(devGrayImage, devSmoothImage, filterSize ,newWidth, height, width);
	  cudaDeviceSynchronize();
	  kernelTime.stop();
	}
	else if (2 == kernelToLaunch)
	{
	  kernelTime.start();
	  kernel_smooth_tweak<72*20><<<dimGrid, dimBlock>>>(devGrayImage, devSmoothImage, newWidth, height, width);
	  cudaDeviceSynchronize();
	  kernelTime.stop();
	}
	else if (3 == kernelToLaunch)
	{
	  kernelTime.start();
	  kernel_smooth_perfect<tileDim><<<dimGrid, dimBlock>>>(devGrayImage, devSmoothImage,newWidth, height, width);
	  cudaDeviceSynchronize();
	  kernelTime.stop();
	}
	else
	{
	  kernelTime.start();
	  kernel_smooth_convolution<tileDim * tileDim><<<dimGrid, dimBlock>>>(devGrayImage, devSmoothImage,newWidth, height, width);
	  cudaDeviceSynchronize();
	  kernelTime.stop();
	}

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess )
	{
	  cerr << "Uh, the kernel had some kind of issue." << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devSmoothImage);
	  return;
	}

	memoryTime.start();

	if (true == padding)
	{
	  devRetVal = cudaMemcpy(newSmoothImage, devSmoothImage, iImageSize, cudaMemcpyDeviceToHost);
	  UnpadImage(newSmoothImage, newWidth, width, height, smoothImage);
	}
	else
	{
	  devRetVal = cudaMemcpy(smoothImage, devSmoothImage, iImageSize, cudaMemcpyDeviceToHost);
	}
	if (cudaSuccess != devRetVal)
	{
	  cout << "cannot copy memory" << endl;
	  cudaFree(devGrayImage);
	  cudaFree(devSmoothImage);

	  return;
	}
	memoryTime.stop();

	cudaFree(devGrayImage);
	cudaFree(devSmoothImage);
	//cudaFree(devFilter);
	if (padding == true)
	{
	  delete []newSmoothImage;
	  delete []newGrayImage;
	}

	#else
	// Kernel
	kernelTime.start();
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			unsigned int filterItem = 0;
			float filterSum = 0.0f;
			float smoothPix = 0.0f;

			for ( int fy = y - 2; fy < y + 3; fy++ ) {
				for ( int fx = x - 2; fx < x + 3; fx++ ) {
					if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) {
						filterItem++;
						continue;
					}

					smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
					filterSum += filter[filterItem];
					filterItem++;
				}
			}

			smoothPix /= filterSum;
			smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	#endif
	
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (kernel): \t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (memory): \t" << memoryTime.getElapsed() << " seconds." << endl;
}

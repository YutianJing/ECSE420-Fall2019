
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lodepng.h"
#define NUM_THREADS 256
__global__ void rectify(unsigned char* image, unsigned char* new_image, int round, int numThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int magicNumber = (numThreads / 1024) * 1024;
	int magicNumber = NUM_THREADS;

	if (i < numThreads) {
		
		if (image[(round * magicNumber + i) * 4] >= 127) // R
			new_image[(round * magicNumber + i) * 4] = image[(round * magicNumber + i) * 4];
		else new_image[(round * magicNumber + i) * 4] = 127;

		if (image[(round * magicNumber + i) * 4 + 1] >= 127) // G
			new_image[(round * magicNumber + i) * 4 + 1] = image[(round * magicNumber + i) * 4 + 1];
		else new_image[(round * magicNumber + i) * 4 + 1] = 127;

		if (image[(round * magicNumber + i) * 4 + 2] >= 127) // B
			new_image[(round * magicNumber + i) * 4 + 2] = image[(round * magicNumber + i) * 4 + 2];
		else new_image[(round * magicNumber + i) * 4 + 2] = 127;

		new_image[(round * magicNumber + i) * 4 + 3] = image[(round * magicNumber + i) * 4 + 3]; // A
	}
}

__global__ void pool(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height, int round, int numThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int magicNumber = (numThreads / 1024) * 1024;
	int magicNumber = NUM_THREADS;
	unsigned char tl, tr, bl, br, max;
	unsigned offset;

	if (i < numThreads) {
		for (int k = 0; k < 4; k++) {
			offset = round * magicNumber * 2 + i * 2;
			offset += width * (offset / width);

			tl = image[(offset) * 4 + k];
			tr = image[(offset + 1) * 4 + k];
			bl = image[(offset + width) * 4 + k];
			br = image[(offset + width + 1) * 4 + k];

			max = 0;

			if (tl > max) max = tl;
			if (tr > max) max = tr;
			if (bl > max) max = bl;
			if (br > max) max = br;

			new_image[(round * magicNumber + i) * 4 + k] = max;
		}
	}
}

void imageRectify(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*) malloc(width * height * 4 * sizeof(unsigned char));

	// sequential way of rectifying
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) {

	//		if (image[4 * width * i + 4 * j] >= 127)
	//			new_image[4 * width * i + 4 * j] = image[4 * width * i + 4 * j]; // R
	//		else new_image[4 * width * i + 4 * j] = 127;

	//		if (image[4 * width * i + 4 * j + 1] >= 127)
	//			new_image[4 * width * i + 4 * j + 1] = image[4 * width * i + 4 * j + 1]; // G
	//		else new_image[4 * width * i + 4 * j + 1] = 127;

	//		if (image[4 * width * i + 4 * j + 2] >= 127)
	//			new_image[4 * width * i + 4 * j + 2] = image[4 * width * i + 4 * j + 2]; // B
	//		else new_image[4 * width * i + 4 * j + 2] = 127;

	//		new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A

	//	}
	//}
	//clock_t start, end;
	//start = clock();
	////////////////////////////////////////////////////////////////////////////////
	// parallel way of rectifying
	cudaSetDevice(0);

	unsigned char* image_dev;
	cudaMallocManaged((void**)&image_dev, width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, width * height * 4 * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		image_dev[i] = image[i];
		new_image[i] = 0;
	}

	int round = 0;
	while (round < width * height / NUM_THREADS) {
		rectify << <(int)ceil((NUM_THREADS + 1023) / 1024), 1024 >> > (image_dev, new_image, round, NUM_THREADS);
		round++;
	}
		rectify << <(int)ceil((NUM_THREADS + 1023) / 1024), (height * width) % 1024 >> > (image_dev, new_image, round, NUM_THREADS);

	cudaDeviceSynchronize();
	//cudaFree(image); cudaFree(new_image); cudaFree(width_p); cudaFree(height_p); cudaFree(image_dev); cudaFree(new_image_dev);
	//end = clock();
	//printf("time=%f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
	//////////////////////////////////////////////////////////////////////////////////

	lodepng_encode32_file(output_filename, new_image, width, height);
	cudaFree(image); cudaFree(new_image); cudaFree(image_dev);
	free(image);
	//free(new_image);
	//free(image_dev);
}

void imagePooling(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));

	clock_t start, end;
	start = clock();
	////////////////////////////////////////////////////////////////////////////////
	// parallel way of pooling
	cudaSetDevice(0);

	unsigned char* image_dev;
	cudaMallocManaged((void**)&image_dev, width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, width * height * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		image_dev[i] = image[i];
	}
	for (int i = 0; i < width * height; i++) new_image[i] = 0;

	int round = 0;
	while (round < width * height / NUM_THREADS / 4) {
		pool << <(int)ceil((NUM_THREADS + 1023) / 1024), 1024 >> > (image_dev, new_image, width, height, round, NUM_THREADS);
		round++;
	}
		pool << <(int)ceil((NUM_THREADS + 1023) / 1024), (height * width) % 1024 >> > (image_dev, new_image, width, height, round, NUM_THREADS);

	cudaDeviceSynchronize();
	end = clock();
	printf("time=%f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
	//////////////////////////////////////////////////////////////////////////////////

	lodepng_encode32_file(output_filename, new_image, width / 2, height / 2);
	cudaFree(image); cudaFree(new_image); cudaFree(image_dev);
	free(image);
	//free(new_image);
	//free(image_dev);
}

int main()
{

	char* input_filename = "test.png";
	char* output_filename_rectify = "test rectify.png";
	char* output_filename_pooling = "test pooling.png";

	imageRectify(input_filename, output_filename_rectify);
	imagePooling(input_filename, output_filename_pooling);

	return 0;

}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lodepng.h"
#include "wm.h"
#include "A_32.h"
#include "A_512.h"
#include "A_1024.h"
#include "b_32.h"
#include "b_512.h"
#include "b_1024.h"
#define NUM_THREADS 1024
#define wmDimension 3

__global__ void convolve(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height, int round, float* wm_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float patch[wmDimension * wmDimension];
	float sum;
	unsigned offset;
	//printf("%d\n", round);
	if (i < NUM_THREADS) {
		for (int k = 0; k < 4; k++) {
			offset = round * NUM_THREADS + i;
			//offset += width * (offset / width);
			
			if (offset % width < (width - wmDimension + 1) && offset < width * (height - wmDimension + 1)) {
				sum = 0;
				for (int j = 0; j < wmDimension * wmDimension; j++) {
					patch[j] = image[(offset + width * (j / wmDimension) + (j - wmDimension * (j / wmDimension))) * 4 + k];

					patch[j] = patch[j] * wm_dev[j];

					sum += patch[j];
				}
				if (sum < 0.0) sum = 0;
				if (sum > 255.0) sum = 255;
				if (k == 3) sum = 255;

				new_image[offset * 4 + k] = sum;
			}
		}
	}
}

void imageConvolution(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc((width - wmDimension + 1) * (height - wmDimension + 1) * sizeof(unsigned char));

	clock_t start, end;
	start = clock();
	////////////////////////////////////////////////////////////////////////////////
	cudaSetDevice(0);

	unsigned char* image_dev;
	cudaMallocManaged((void**)&image_dev, width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, (width - wmDimension + 1) * (height - wmDimension + 1) * 4 * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		image_dev[i] = image[i];
	}
	for (int i = 0; i < (width - wmDimension + 1) * (height - wmDimension + 1) * 4; i++) new_image[i] = 0;
	
	float* wm_dev;
	cudaMallocManaged((void**)&wm_dev, wmDimension * wmDimension * sizeof(float));
	for (int i = 0; i < wmDimension; i++) {
		for (int j = 0; j < wmDimension; j++) {
			if (wmDimension == 3) wm_dev[i * wmDimension + j] = w3[i][j];
			if (wmDimension == 5) wm_dev[i * wmDimension + j] = w5[i][j];
			if (wmDimension == 7) wm_dev[i * wmDimension + j] = w7[i][j];
		}
	}

	int round = 0;
	int numBlocks = (int)ceil(((double)NUM_THREADS + (double)1023) / (double)1024);
	while (round < (width - wmDimension + 1) * (height - wmDimension + 1) / NUM_THREADS) {
		convolve << <numBlocks, 1024 >> > (image_dev, new_image, width, height, round, wm_dev);
		round++;
	}
	//convolve << <numBlocks, (height * width) % 1024 >> > (image_dev, new_image, width, height, round, w3, w5, w7);

	cudaDeviceSynchronize();
	////////////////////////////////////////////////////////////////////////////////
	end = clock();
	printf("time=%f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
	
	lodepng_encode32_file(output_filename, new_image, (width - wmDimension + 1), (height - wmDimension + 1));
	cudaFree(image); cudaFree(new_image); cudaFree(image_dev); cudaFree(wm_dev);
	free(image);
	//free(new_image);
	//free(image_dev);
}

int main()
{

	char* input_filename = "test.png";
	char* output_filename_convolution3 = "test convolve 3x3.png";
	char* output_filename_convolution5 = "test convolve 5x5.png";
	char* output_filename_convolution7 = "test convolve 7x7.png";

	if (wmDimension == 3) imageConvolution(input_filename, output_filename_convolution3);
	if (wmDimension == 5) imageConvolution(input_filename, output_filename_convolution5);
	if (wmDimension == 7) imageConvolution(input_filename, output_filename_convolution7);
	
	return 0;

}

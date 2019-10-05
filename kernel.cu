
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"
#define NUM_THREADS 1

__global__ void rectifyKernel(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height, int counter, int threads)
{
	int i = threadIdx.x;

	if (i < threads) {
		if (image[(counter * threads + i) * 4] >= 127) // R
			new_image[(counter * threads + i) * 4] = image[(counter * threads + i) * 4];
		else new_image[(counter * threads + i) * 4] = 127;

		if (image[(counter * threads + i) * 4 + 1] >= 127) // G
			new_image[(counter * threads + i) * 4 + 1] = image[(counter * threads + i) * 4 + 1];
		else new_image[(counter * threads + i) * 4 + 1] = 127;

		if (image[(counter * threads + i) * 4 + 2] >= 127) // B
			new_image[(counter * threads + i) * 4 + 2] = image[(counter * threads + i) * 4 + 2];
		else new_image[(counter * threads + i) * 4 + 2] = 127;

		new_image[(counter * threads + i) * 4 + 3] = image[(counter * threads + i) * 4 + 3]; // A
	}
}

void rectify(char* input_filename, char* output_filename)
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
	
	////////////////////////////////////////////////////////////////////////////////
	// parallel way of rectifying
	cudaSetDevice(0);

	unsigned char* image_dev;
	cudaMallocManaged((void**)&image_dev, NUM_THREADS * width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, NUM_THREADS * width * height * 4 * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		image_dev[i] = image[i];
		new_image[i] = 0;
	}

	for (int counter = 0; counter < width * height / NUM_THREADS; counter++) {
		rectifyKernel << <1, NUM_THREADS >> > (image_dev, new_image, width, height, counter, NUM_THREADS);
	}

	cudaDeviceSynchronize();
	//cudaFree(image); cudaFree(new_image); cudaFree(width_p); cudaFree(height_p); cudaFree(image_dev); cudaFree(new_image_dev);
	//////////////////////////////////////////////////////////////////////////////////

	lodepng_encode32_file(output_filename, new_image, width, height);
	cudaFree(image); cudaFree(new_image); cudaFree(image_dev);
	free(image);
	//free(new_image);
	//free(image_dev);
}

int main()
{

	char* input_filename = "test.png";
	char* output_filename = "test rectify.png";

	rectify(input_filename, output_filename);

	return 0;

}

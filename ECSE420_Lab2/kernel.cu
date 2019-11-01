
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

#include "A_3.h"
#include "b_3.h"
#define NUM_THREADS 1024
#define wmSIZE 3
#define matrixSIZE 3

float AInv[matrixSIZE][matrixSIZE];

__global__ void convolve(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height, int round, float* wm_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float patch[wmSIZE * wmSIZE];
	float sum;
	unsigned offset;
	//printf("%d\n", round);
	if (i < NUM_THREADS) {
		for (int k = 0; k < 4; k++) {
			offset = round * NUM_THREADS + i;
			//offset += width * (offset / width);
			//offset -= offset % (width - wmSIZE);

			if ((offset % width) < (width - wmSIZE + 1) && offset < width * (height - wmSIZE + 1)) {
				sum = 0;
				for (int j = 0; j < wmSIZE * wmSIZE; j++) {
					patch[j] = image[(offset + width * (j / wmSIZE) + (j - wmSIZE * (j / wmSIZE))) * 4 + k];

					patch[j] = patch[j] * wm_dev[j];

					sum += patch[j];
				}
				if (sum < 0.0) sum = 0;
				if (sum > 255.0) sum = 255;
				if (k == 3) sum = image[offset * 4 + k];

				new_image[(offset - (offset / width) * (wmSIZE - 1)) * 4 + k] = sum;
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
	new_image = (unsigned char*)malloc((width - wmSIZE + 1) * (height - wmSIZE + 1) * sizeof(unsigned char));

	clock_t start, end;
	start = clock();
	////////////////////////////////////////////////////////////
	cudaSetDevice(0);

	unsigned char* image_dev;
	cudaMallocManaged((void**)&image_dev, width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, (width - wmSIZE + 1) * (height - wmSIZE + 1) * 4 * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		image_dev[i] = image[i];
	}
	for (int i = 0; i < (width - wmSIZE + 1) * (height - wmSIZE + 1) * 4; i++) new_image[i] = 0;
	
	float* wm_dev;
	cudaMallocManaged((void**)&wm_dev, wmSIZE * wmSIZE * sizeof(float));
	for (int i = 0; i < wmSIZE; i++) {
		for (int j = 0; j < wmSIZE; j++) {
			if (wmSIZE == 3) wm_dev[i * wmSIZE + j] = w3[i][j];
			if (wmSIZE == 5) wm_dev[i * wmSIZE + j] = w5[i][j];
			if (wmSIZE == 7) wm_dev[i * wmSIZE + j] = w7[i][j];
		}
	}

	int round = 0;
	int numBlocks = (int)ceil(((double)NUM_THREADS + (double)1023) / (double)1024);
	while (round < width * height / NUM_THREADS) {
		convolve << <numBlocks, 1024 >> > (image_dev, new_image, width, height, round, wm_dev);
		round++;
	}
	//convolve << <numBlocks, (height * width) % 1024 >> > (image_dev, new_image, width, height, round, wm_dev);

	cudaDeviceSynchronize();
	////////////////////////////////////////////////////////////
	end = clock();
	printf("time=%f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
	
	lodepng_encode32_file(output_filename, new_image, (width - wmSIZE + 1), (height - wmSIZE + 1));
	cudaFree(image); cudaFree(new_image); cudaFree(image_dev); cudaFree(wm_dev);
	free(image);
	//free(new_image);
	//free(image_dev);
}

float getDet(float inputMatrix[matrixSIZE][matrixSIZE], int n)
{
	if (n == 1) return inputMatrix[0][0];
	float ans = 0;
	float temp[matrixSIZE][matrixSIZE];
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n - 1; j++) {
			for (k = 0; k < n - 1; k++) {
				temp[j][k] = inputMatrix[j + 1][(k >= i) ? k + 1 : k];
			}
		}
		float t = getDet(temp, n - 1);
		if (i % 2 == 0) {
			ans += inputMatrix[0][i] * t;
		}
		else {
			ans -= inputMatrix[0][i] * t;
		}
	}
	return ans;
}

void getAStar(float inputMatrix[matrixSIZE][matrixSIZE], int n, float ans[matrixSIZE][matrixSIZE])
{
	if (n == 1) {
		ans[0][0] = 1;
		return;
	}
	int i, j, k, t;
	float temp[matrixSIZE][matrixSIZE];
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < n - 1; k++) {
				for (t = 0; t < n - 1; t++) {
					temp[k][t] = inputMatrix[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
				}
			}
			ans[j][i] = getDet(temp, n - 1);
			if ((i + j) % 2 == 1) {
				ans[j][i] = -ans[j][i];
			}
		}
	}
}

float inverse(float inputMatrix[matrixSIZE][matrixSIZE])
{
	float AStar[matrixSIZE][matrixSIZE];
	float det = getDet(inputMatrix, matrixSIZE);
	if (det == 0) { printf("The input matrix can not be transformed!\n"); }
	else {
		getAStar(inputMatrix, matrixSIZE, AStar);
		for (int i = 0; i < matrixSIZE; i++) {
			for (int j = 0; j < matrixSIZE; j++) {
				AInv[i][j] = AStar[i][j] / det;
			}
		}
	}
	return 0;
}

__global__ void multiply_and_add(float* x, float* AInv_dev, float* b_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_THREADS) {
		for (int j = 0; j < matrixSIZE; j++) {
			x[i] += AInv_dev[i * matrixSIZE + j % matrixSIZE] * b_dev[j];
		}
	}
}

void printArray(float* a, int n) { for (int i = 0; i < n; i++) { printf("x[%d] = %f\n", i, a[i]); } }

void solve_Ax_equals_b()
{	
	////////////////////////////////////////////////// Question 2: Solve Ax = b
	float A[matrixSIZE][matrixSIZE];
	for (int i = 0; i < matrixSIZE; i++) {
		for (int j = 0; j < matrixSIZE; j++) {
			if (matrixSIZE == 3) A[i][j] = A_3[i][j];
			if (matrixSIZE == 32) A[i][j] = A_32[i][j];
			if (matrixSIZE == 512) A[i][j] = A_512[i][j];
			if (matrixSIZE == 1024) A[i][j] = A_1024[i][j];
		}
	}
	inverse(A); // now the inverse is stored in AInv

	cudaSetDevice(0);

	float* x;
	x = (float*)malloc(matrixSIZE * sizeof(float));
	cudaMallocManaged((void**)&x, matrixSIZE * sizeof(float));
	for (int i = 0; i < matrixSIZE; i++) x[i] = 0;

	float* AInv_dev;
	cudaMallocManaged((void**)&AInv_dev, matrixSIZE * matrixSIZE * sizeof(float));
	for (int i = 0; i < matrixSIZE; i++) {
		for (int j = 0; j < matrixSIZE; j++) {
			AInv_dev[i * matrixSIZE + j] = AInv[i][j];
		}
	}

	float* b_dev;
	cudaMallocManaged((void**)&b_dev, matrixSIZE * sizeof(float));
	for (int i = 0; i < matrixSIZE; i++) {
		if (matrixSIZE == 3) b_dev[i] = b_3[i][0];
		if (matrixSIZE == 32) b_dev[i] = b_32[i][0];
		if (matrixSIZE == 512) b_dev[i] = b_512[i][0];
		if (matrixSIZE == 1024) b_dev[i] = b_1024[i][0];
	}
	
	multiply_and_add << <1, matrixSIZE >> > (x, AInv_dev, b_dev);

	cudaDeviceSynchronize();

	printArray(x, matrixSIZE);

	cudaFree(AInv_dev); cudaFree(b_dev); cudaFree(x);
}

int main()
{
	char* input_filename = "test.png";
	char* output_filename_convolution3 = "test convolve 3x3.png";
	char* output_filename_convolution5 = "test convolve 5x5.png";
	char* output_filename_convolution7 = "test convolve 7x7.png";

	/*if (wmSIZE == 3) imageConvolution(input_filename, output_filename_convolution3);
	if (wmSIZE == 5) imageConvolution(input_filename, output_filename_convolution5);
	if (wmSIZE == 7) imageConvolution(input_filename, output_filename_convolution7);*/
	
	solve_Ax_equals_b();

	return 0;
}

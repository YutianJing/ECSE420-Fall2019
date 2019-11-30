
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define Rho 0.5
#define Eta 0.0002
#define G 0.75
#define N 512
#define DECOMPOSITION_PER_ROW 51 // row decomposition, furthermore devide a row into multiple pieces, minimum is 1
#define ELEMENT_PER_THREAD (N - 2) / DECOMPOSITION_PER_ROW // one thread handles all the elements in a single piece
#define NUM_THREADS ((N - 2) * (N - 2)) / ((N - 2) / DECOMPOSITION_PER_ROW)
#define NUM_THREADS_EDGE 4 * DECOMPOSITION_PER_ROW
#define NUM_BLOCKS (NUM_THREADS + 1024 - NUM_THREADS % 1024) / 1024
#define NUM_BLOCKS_EDGE (NUM_THREADS_EDGE + 1024 - NUM_THREADS_EDGE % 1024) / 1024

void iteration_sequential(float u[N * N], float u1[N * N], float u2[N * N])
{
	clock_t start, end;
	
	for (int i = 1; i <= N - 2; i++) {
		for (int j = 1; j <= N - 2; j++) {
			u[i * N + j] = Rho * (u1[(i - 1) * N + j] + u1[(i + 1) * N + j] + u1[i * N + (j - 1)] + u1[i * N + (j + 1)] - 4 * u1[i * N + j]);
			u[i * N + j] += 2 * u1[i * N + j] - (1 - Eta) * u2[i * N + j];
			u[i * N + j] /= (1 + Eta);
		}
	}
	
	for (int i = 1; i <= N - 2; i++) {
		u[0 * N + i]		= G * u[1 * N + i];
		u[(N - 1) * N + i]	= G * u[(N - 2) * N + i];
		u[i * N + 0]		= G * u[i * N + 1];
		u[i * N + (N - 1)]	= G * u[i * N + (N - 2)];
	}
	
	u[0 * N + 0]				= G * u[1 * N + 0];
	u[(N - 1) * N + 0]			= G * u[(N - 2) * N + 0];
	u[0 * N + (N - 1)]			= G * u[0 * N + (N - 2)];
	u[(N - 1) * N + (N - 1)]	= G * u[(N - 1) * N + (N - 2)];
	
	for (int i = 0; i < N * N; i++) {
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	/*for (int i = 0; i < N * N; i++) {
			printf("(%d,%d): ", i / N, i % N);
			printf("%.6f ", u[i]);
			if ((i + 1) % N == 0) printf("\n");
	}*/
}

__global__ void iteration_parallel_central(float* u_dev, float* u1_dev, float* u2_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_THREADS) {
		for (int j = 0; j < ELEMENT_PER_THREAD; j++) {
			// index conversion from 2D to flat: u[i][j] = u[x * N + y]
			// int x = (i * (N - 2) + j) / (N - 2) + 1;
			// int y = (i * (N - 2) + j) % (N - 2) + 1;
			int x = (i / DECOMPOSITION_PER_ROW) + 1;
			int y = (i % DECOMPOSITION_PER_ROW) * ELEMENT_PER_THREAD + 1 + j;

			u_dev[x * N + y] = Rho * (u1_dev[(x - 1) * N + y] + u1_dev[(x + 1) * N + y] + u1_dev[x * N + (y - 1)] + u1_dev[x * N + (y + 1)] - 4 * u1_dev[x * N + y]);
			u_dev[x * N + y] += 2 * u1_dev[x * N + y] - (1 - Eta) * u2_dev[x * N + y];
			u_dev[x * N + y] /= (1 + Eta);
		}
	}
}

__global__ void iteration_parallel_edge(float* u_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 0 && i < NUM_THREADS_EDGE / 4) {
		for (int j = 0; j < ELEMENT_PER_THREAD; j++) {
			// index conversion from 2D to flat: u[i][j] = u[x * N + y]
			int offset = (i % DECOMPOSITION_PER_ROW) * ELEMENT_PER_THREAD + 1 + j;
			u_dev[0 * N + offset] = G * u_dev[1 * N + offset];
		}
	}
	if (i >= NUM_THREADS_EDGE / 4 && i < NUM_THREADS_EDGE / 4 * 2) {
		for (int j = 0; j < ELEMENT_PER_THREAD; j++) {
			// index conversion from 2D to flat: u[i][j] = u[x * N + y]
			int offset = (i % DECOMPOSITION_PER_ROW) * ELEMENT_PER_THREAD + 1 + j;
			u_dev[(N - 1) * N + offset] = G * u_dev[(N - 2) * N + offset];
		}
	}
	if (i >= NUM_THREADS_EDGE / 4 * 2 && i < NUM_THREADS_EDGE / 4 * 3) {
		for (int j = 0; j < ELEMENT_PER_THREAD; j++) {
			// index conversion from 2D to flat: u[i][j] = u[x * N + y]
			int offset = (i % DECOMPOSITION_PER_ROW) * ELEMENT_PER_THREAD + 1 + j;
			u_dev[offset * N + 0] = G * u_dev[offset * N + 1];
		}
	}
	if (i >= NUM_THREADS_EDGE / 4 * 3 && i < NUM_THREADS_EDGE) {
		for (int j = 0; j < ELEMENT_PER_THREAD; j++) {
			// index conversion from 2D to flat: u[i][j] = u[x * N + y]
			int offset = (i % DECOMPOSITION_PER_ROW) * ELEMENT_PER_THREAD + 1 + j;
			u_dev[offset * N + (N - 1)] = G * u_dev[offset * N + (N - 2)];
		}
	}
}

__global__ void iteration_parallel_update(float* u_dev, float* u1_dev, float* u2_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N * N) {
		u2_dev[i] = u1_dev[i];
		u1_dev[i] = u_dev[i];
	}
}

int main(int argc, char* argv[])
{
	printf("argc = %d, argv = %s\n", argc, argv[1]);
	int T = atoi(argv[1]);
	
	float* u = (float*)malloc(N * N * sizeof(float));
	float* u1 = (float*)malloc(N * N * sizeof(float));
	float* u2 = (float*)malloc(N * N * sizeof(float));
	for (int i = 0; i < N * N; i++) {
		u[i] = 0.f;
		u1[i] = 0.f;
		u2[i] = 0.f;
	}

	//////////////////////////////
	// sequential approach
	clock_t start, end;
	start = clock();
	for (int i = 0; i < T; i++) {
		if (i == 0) u1[(N / 2) * N + (N / 2)] += 1;
		printf("Iteration: %d,    ", i);
		iteration_sequential(u, u1, u2);
		printf("(%d,%d): ", N / 2, N / 2);
		printf("%.6f\n", u[(N / 2) * N + (N / 2)]);
	}
	end = clock();
	printf("Time spent = %f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
	//////////////////////////////

	printf("\n\n");

	//////////////////////////////
	// parallel approach
	cudaSetDevice(0);

	float* u_dev;
	float* u1_dev;
	float* u2_dev;
	cudaMallocManaged((void**)&u_dev, N * N * sizeof(float));
	cudaMallocManaged((void**)&u1_dev, N * N * sizeof(float));
	cudaMallocManaged((void**)&u2_dev, N * N * sizeof(float));
	for (int i = 0; i < N * N; i++) {
		u_dev[i] = 0.f;
		u1_dev[i] = 0.f;
		u2_dev[i] = 0.f;
	}

	start = clock();
	for (int i = 0; i < T; i++) {

		if (i == 0) u1_dev[(N / 2) * N + (N / 2)] += 1;

		printf("Iteration: %d,    ", i);
		
		iteration_parallel_central << <NUM_BLOCKS, 1024 >> > (u_dev, u1_dev, u2_dev);
		cudaDeviceSynchronize();
		
		/*cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
		}*/
		
		iteration_parallel_edge << <NUM_BLOCKS_EDGE, 1024 >> > (u_dev);
		cudaDeviceSynchronize();
		
		u_dev[0 * N + 0]				= G * u_dev[1 * N + 0];
		u_dev[(N - 1) * N + 0]			= G * u_dev[(N - 2) * N + 0];
		u_dev[0 * N + (N - 1)]			= G * u_dev[0 * N + (N - 2)];
		u_dev[(N - 1) * N + (N - 1)]	= G * u_dev[(N - 1) * N + (N - 2)];
		
		iteration_parallel_update << <((N * N) + 1024 - (N * N) % 1024) / 1024, 1024 >> > (u_dev, u1_dev, u2_dev);
		cudaDeviceSynchronize();

		printf("(%d,%d): ", N / 2, N / 2);
		printf("%.6f\n", u_dev[(N / 2) * N + (N / 2)]);
	}
	end = clock();
	printf("Time spent = %f\n", (double)(end - start) / (double)CLOCKS_PER_SEC);

	cudaFree(u_dev); cudaFree(u1_dev); cudaFree(u2_dev);
	//////////////////////////////

    return 0;
}

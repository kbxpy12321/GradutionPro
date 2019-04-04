#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include "Matrix.h"
#include "MatrixCalculation.h"
#include "CudaMatrixCalculation.cuh"
#define CUDAMAXTHREADPERBLOCK 1024
#define CUDAMAXBLOCK 65536

using namespace std;
__global__ void kernelAddConstant(int *g_a, const int b)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_a[idx] += b;
}

//__global__ void kernelMatrixMul(int *matrixA, int *matrixB, int *matrixC, int sameside) {
//	
//	int col = sizeof(matrixB) / sizeof(matrixB[0]) / sameside;
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	matrixC[idx] = 0;
//	for (int i = 0; i < sameside; i++) {
//		matrixC[idx] += matrixA[blockIdx.x * sameside + i] * matrixB[i * col + threadIdx.x];
//	}
//}



__global__ void kernelMatrixAdd(Matrix *matrixA, Matrix *matrixB) {

}


template<typename matrixAT, typename matrixBT, typename matrixCT>
__global__ void kernelMatrixMul(matrixAT *matrixA, matrixBT *matrixB, matrixCT *matrixC, int row, int sameside, int col) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= row * col)
		return;
	/*if (blockIdx.x == 0 && threadIdx.x == 0) {
		printf("sizeof matrixb: %d", sizeof(matrixB));
		printf("device: \n");
		for (int i = 0; i < 16; i++) {
			printf("%d %d\n", matrixA[i], matrixB[i]);
		}
		printf("\n");
	}

	printf("device: col: %d blockidx: %d blockDimx: %d threadIdx: %d\n", col, blockIdx.x, blockDim.x, threadIdx.x);
	printf("\n");*/

	int curRow, curCol;
	curRow = idx / col;
	curCol = idx % col;

	matrixC[idx] = 0;
	for (int i = 0; i < sameside; i++) {
		matrixC[idx] += matrixA[curRow * sameside + i] * matrixB[i * col + curCol];
		//printf("%d %d\n", matrixA[blockIdx.x * sameside + i], matrixB[i * col + threadIdx.x]);
	}
};


template<typename matrixCT>
__global__ void kernelTestRes(matrixCT *matrixRes, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%d ", matrixRes[i * col + j]);
		}
		printf("\n");
	}
}

template <typename matrixAT, typename matrixBT, typename matrixCT>
Matrix *matrixMul(Matrix *matrixA, Matrix* matrixB) {
	if (matrixA == NULL || matrixB == NULL) {
		return nullptr;
	}
	int fullLen = matrixA->getRow() * matrixB->getCol();
	if (fullLen > CUDAMAXTHREADPERBLOCK * CUDAMAXBLOCK) {
		return nullptr;
	}
	int row = matrixA->getRow();
	int col = matrixB->getCol();
	int sameSide = matrixA->getCol();
	
	int sizeA = row * matrixA->getCol();
	int sizeB = col * matrixB->getRow();
	int finalType = MatrixCalculation::matrixTypeDecision(matrixA->getType(), matrixB->getType());

	int calRow, calCol;
	calCol = CUDAMAXTHREADPERBLOCK;
	calRow = fullLen / CUDAMAXTHREADPERBLOCK;
	if (calRow < 1) {
		calRow = 1;
		calCol = col * row;
	}
	else if (fullLen % calCol != 0) {
		calRow++;
	}

	matrixAT* dev_A;
	matrixAT* host_A;
	host_A = (matrixAT*)(matrixA->returnVectorData());
	cudaMalloc((void **)&dev_A, sizeA * sizeof(matrixAT));
	cudaMemcpy(dev_A, host_A, sizeA * sizeof(matrixAT), cudaMemcpyHostToDevice);

	matrixBT* dev_B;
	matrixBT* host_B;
	host_B = (matrixBT*)(matrixB->returnVectorData());
	cudaMalloc((void **)&dev_B, sizeB * sizeof(matrixBT));
	cudaMemcpy(dev_B, host_B, sizeB * sizeof(matrixBT), cudaMemcpyHostToDevice);

	/*printf("host: \n");
	for (int i = 0; i < 16; i++) {
		printf("%d %d\n", host_A[i], host_B[i]);
	}
	printf("\n");*/

	matrixCT *dev_C, *host_C;
	cudaMalloc((void **)&dev_C, fullLen * sizeof(matrixCT));

	kernelMatrixMul<matrixAT, matrixBT, matrixCT> << <calRow, calCol >> > (dev_A, dev_B, dev_C, row, sameSide, col);

	Matrix *matrixRes = new Matrix(matrixA->getRow(), matrixB->getCol(), finalType);
	matrixRes->initVectorSpace();
	host_C = (matrixCT* )(matrixRes->returnVectorData());
	cudaMemcpy(host_C, dev_C, fullLen * sizeof(matrixCT), cudaMemcpyDeviceToHost);


	//kernelTestRes<matrixCT> << <1, 1 >> > (dev_C, 4, 4);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	return matrixRes;
};

extern "C" Matrix *matrixMulByCuda(Matrix *matrixA, Matrix *matrixB) {
	int matrixTypeA = matrixA->getType();
	int matrixTypeB = matrixB->getType();
	int greaterType = MatrixCalculation::matrixTypeDecision(matrixTypeA, matrixTypeB);
	if (greaterType == matrixTypeA && greaterType == matrixTypeB) {
		if (greaterType == INTEGER) {
			return matrixMul<int, int, int>(matrixA, matrixB);
		}
		else if (greaterType == FLOAT) {
			return matrixMul<float, float, float>(matrixA, matrixB);
		}
		else if (greaterType == DOUBLE) {
			return matrixMul<double, double, double>(matrixA, matrixB);
		}
		else if (greaterType == LONGLONG) {
			return matrixMul<long long, long long, long long>(matrixA, matrixB);
		}
		return nullptr;
	}
	if (greaterType == matrixTypeA) {
		if (greaterType == DOUBLE) {
			if (matrixTypeB == LONGLONG) {
				return matrixMul<double, long long, double>(matrixA, matrixB);
			}
			else if (matrixTypeB == FLOAT) {
				return matrixMul<double, float, double>(matrixA, matrixB);
			}
			else if (matrixTypeB == INTEGER) {
				return matrixMul<double, int, double>(matrixA, matrixB);
			}
		}
		else if (greaterType == FLOAT) {
			return matrixMul<float, int, float>(matrixA, matrixB);
		}
		else if (greaterType == LONGLONG) {
			return matrixMul<long long, int, long long>(matrixA, matrixB);
		}
		return nullptr;
	}
	if (greaterType == matrixTypeB) {
		if (greaterType == DOUBLE) {
			if (matrixTypeA == LONGLONG) {
				return matrixMul<long long, double, double>(matrixA, matrixB);
			}
			else if (matrixTypeA == FLOAT) {
				return matrixMul<float, double, double>(matrixA, matrixB);
			}
			else if (matrixTypeA == INTEGER) {
				return matrixMul<int, double, double>(matrixA, matrixB);
			}
		}
		else if (greaterType == FLOAT) {
			return matrixMul<int, float, float>(matrixA, matrixB);
		}
		else if (greaterType == LONGLONG) {
			return matrixMul<int, long long, long long>(matrixA, matrixB);
		}
		return nullptr;
	}
	if (matrixTypeA == LONGLONG)
		return matrixMul<long long, float, double>(matrixA, matrixB);
	if (matrixTypeB == LONGLONG)
		return matrixMul<float, long long, double>(matrixA, matrixB);
	return nullptr;
}

#define ttN 64000000

__global__ void kernelTestOpenmp(int *dev_b, int tt){
	for (int i = 0; i < tt; i++) {
		if (dev_b[i] != i) {
			printf("no!!!");
		}
		printf("yes!!!!");
	}
}



extern "C" int testMulCoreSingleGPU(int coreNum) {
	omp_set_num_threads(coreNum);
	int *b = new int[ttN];
	for (int i = 0; i < ttN; i++) {
		b[i] = i;
	}
	int *dev_b;
	cudaMalloc((void **)&dev_b, ttN * sizeof(int));
	cudaMemset(dev_b, 0, ttN * sizeof(int));
	int partB = ttN / coreNum;
#pragma omp parallel
	{
		int coreNow = omp_get_thread_num();
		cudaMemcpy(dev_b + partB * coreNow, b + partB * coreNow, partB * sizeof(int), cudaMemcpyHostToDevice);
	}
	kernelTestOpenmp << <1, 1 >> > (dev_b, ttN);
	return 0;
}

extern "C" int cudaGetGpus() {
	int totalGpuNum = 0;
	cudaGetDeviceCount(&totalGpuNum);
	return totalGpuNum;
}

template <typename matrixAT, typename matrixBT, typename matrixCT>
Matrix *matrixOpenMPMulByCuda(Matrix *matrixA, Matrix *matrixB, int cpuNum, int gpuNum) {
	if (matrixA == NULL || matrixB == NULL) {
		return nullptr;
	}

	int totalGpuNum = 0;
	cudaGetDeviceCount(&totalGpuNum);
	int totalCpuNum = omp_get_num_procs();
	if (totalGpuNum < 1)
	{
		printf("no CUDA capable devices were detected\n");
		return nullptr;
	}

	int fullLen = matrixA->getRow() * matrixB->getCol();
	if (fullLen > CUDAMAXTHREADPERBLOCK * CUDAMAXBLOCK * totalGpuNum) {
		return nullptr;
	}
	int row = matrixA->getRow();
	int col = matrixB->getCol();
	int sameSide = matrixA->getCol();

	int finalType = MatrixCalculation::matrixTypeDecision(matrixA->getType(), matrixB->getType());

	if (gpuNum > totalGpuNum)
		gpuNum = totalGpuNum;
	if (gpuNum > cpuNum)
		cpuNum = gpuNum;

	omp_set_num_threads(cpuNum);

#pragma omp parallel
	{
		int rowDevisionPart = row / cpuNum, colDevisionPart = col / gpuNum;
		int cpuNow = omp_get_thread_num();
		int gpu_id = -1;
		cudaSetDevice(cpuNow % gpuNum);
		cudaGetDevice(&gpu_id);
	}
	return nullptr;
}

extern "C" int testCudaOpenMP() {
	int num_gpus = 0;   // number of CUDA GPUs

	//printf("%s Starting...\n\n", argv[0]);

	/////////////////////////////////////////////////////////////////
	// determine the number of CUDA capable GPUs
	//
	cudaGetDeviceCount(&num_gpus);

	if (num_gpus < 1)
	{
		printf("no CUDA capable devices were detected\n");
		return 1;
	}

	/////////////////////////////////////////////////////////////////
	// display CPU and GPU configuration
	//
	printf("number of host CPUs:\t%d\n", omp_get_num_procs());
	printf("number of CUDA devices:\t%d\n", num_gpus);

	for (int i = 0; i < num_gpus; i++)
	{
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop, i);
		printf("   %d: %s\n", i, dprop.name);
	}

	printf("---------------------------\n");


	/////////////////////////////////////////////////////////////////
	// initialize data
	//
	unsigned int n = num_gpus * 8192;
	unsigned int nbytes = n * sizeof(int);
	int *a = 0;     // pointer to data on the CPU
	int b = 3;      // value by which the array is incremented
	a = (int *)malloc(nbytes);

	if (0 == a)
	{
		printf("couldn't allocate CPU memory\n");
		return 1;
	}

	for (unsigned int i = 0; i < n; i++)
		a[i] = i;


	////////////////////////////////////////////////////////////////
	// run as many CPU threads as there are CUDA devices
	//   each CPU thread controls a different device, processing its
	//   portion of the data.  It's possible to use more CPU threads
	//   than there are CUDA devices, in which case several CPU
	//   threads will be allocating resources and launching kernels
	//   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
	//   Recall that all variables declared inside an "omp parallel" scope are
	//   local to each CPU thread
	//
	omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
	//omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
#pragma omp parallel
	{
		unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int num_cpu_threads = omp_get_num_threads();

		// set and check the CUDA device for this CPU thread
		int gpu_id = -1;
		checkCudaErrors(cudaSetDevice(cpu_thread_id % num_gpus));   // "% num_gpus" allows more CPU threads than GPU devices
		checkCudaErrors(cudaGetDevice(&gpu_id));
		printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

		int *d_a = 0;   // pointer to memory on the device associated with this CPU thread
		int *sub_a = a + cpu_thread_id * n / num_cpu_threads;   // pointer to this CPU thread's portion of data
		unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
		dim3 gpu_threads(128);  // 128 threads per block
		dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

		checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
		checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
		checkCudaErrors(cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
		kernelAddConstant << <gpu_blocks, gpu_threads >> > (d_a, b);

		checkCudaErrors(cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_a));

	}
	printf("---------------------------\n");

	if (cudaSuccess != cudaGetLastError())
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));


	////////////////////////////////////////////////////////////////
	// check the result
	//
	//bool bResult = correctResult(a, n, b);

	if (a)
		free(a); // free CPU memory

	//exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);

	return 1;
}

namespace CudaMatrixCal {
	Matrix *ttt() {
		Matrix *a = new Matrix(3, 3, 1);
		return a;
	}
}

namespace testCuda {
	int testInCuda() {
		int num_gpus = 0;
		cudaGetDeviceCount(&num_gpus);
		return num_gpus;
	}
}

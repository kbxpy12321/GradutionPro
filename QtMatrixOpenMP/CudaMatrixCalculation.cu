#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <Matrix.h>
#include <MatrixCalculation.h>

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

template<typename T>
__global__ void kernelMatrixMul(T *matrixA, T *matrixB, T *matrixC, int sameside) {

	int col = sizeof(matrixB) / sizeof(matrixB[0]) / sameside;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	matrixC[idx] = 0;
	for (int i = 0; i < sameside; i++) {
		matrixC[idx] += matrixA[blockIdx.x * sameside + i] * matrixB[i * col + threadIdx.x];
	}
}

__global__ void kernelMatrixAdd(Matrix *matrixA, Matrix *matrixB) {
	
}



extern "C" Matrix *matrixMulByCuda(Matrix *matrixA, Matrix* matrixB) { 
	if (matrixA == NULL || matrixB == NULL || matrixA->getType() != INTEGER || matrixB->getType() != INTEGER || matrixB->getCol() > 1024) {
		return nullptr;
	}
	
	int row = matrixA->getRow();
	int col = matrixB->getCol();
	int sameSide = matrixA->getCol();
	int fullLen = matrixA->getRow() * matrixB->getCol();
	int sizeA = row * matrixA->getCol();
	int sizeB = col * matrixB->getRow();
	int finalType = MatrixCalculation::matrixTypeDecision(matrixA->getType(), matrixB->getType());
	int tmpType;

	using TYPENOW = int;
	TYPENOW* dev_A;
	TYPENOW* host_A;
	//host_A = (matrixA->returnVector<std::vector<TYPENOW>>()).data();
	cudaMalloc((void **)dev_A, sizeA * sizeof(TYPENOW));
	cudaMemcpy(dev_A, host_A, sizeA * sizeof(TYPENOW), cudaMemcpyHostToDevice);

	//changeTypeNow(matrixB->getType());
	TYPENOW* dev_B;
	TYPENOW* host_B;
	//host_B = (matrixB->returnVector<std::vector<TYPENOW>>()).data();
	cudaMalloc((void **)dev_B, sizeB * sizeof(TYPENOW));
	cudaMemcpy(dev_B, host_B, sizeB * sizeof(TYPENOW), cudaMemcpyHostToDevice);


	//changeTypeNow(finalType);
	TYPENOW *dev_C;
	cudaMalloc((void **)dev_C, fullLen * sizeof(TYPENOW));

	kernelMatrixMul << <row, col>> > (dev_A, dev_B, dev_C, sameSide);

	Matrix *matrixRes = new Matrix();
	//matrixRes->initVectorByArray(dev_C, matrixA->getRow(), matrixB->getCol(), finalType);
	return matrixRes;
}

extern "C" int testInCuda() {
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	return num_gpus;
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


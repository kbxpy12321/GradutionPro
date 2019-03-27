namespace testCuda {
	int testInCuda();
}

namespace CudaMatrixCal {
	template <typename matrixAT, typename matrixBT, typename matrixCT>
	Matrix *matrixMulByCuda(Matrix *matrixA, Matrix* matrixB);
}
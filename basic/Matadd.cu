#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

#define BLOCK_SIZE 16

__global__ void addition(float *A, float *B, float *C, int N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.f;
	sum = A[row*N+col]+B[row*N+col];
	C[row*N+col] = sum;
}

int main(int argc, char *argv[])
{
    // Initialize clock variables
    clock_t cpuClock;
    clock_t gpuClock;

	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	// Restricted to matrices where N = K*BLOCK_SIZE;
	int N,K;
	K = 500;			
	N = K*BLOCK_SIZE;
	
	cout << "Matrix size: " << N << "x" << N << endl << endl;

	// Allocate memory on the host
	float *hA,*hB,*hC;
	hA = new float[N*N];
	hB = new float[N*N];
	hC = new float[N*N];

	// Initialize matrices on the host
	for (int j=0; j<N; j++){
	    for (int i=0; i<N; i++){
	    	hA[j*N+i] = 2.f*(j+i);
			hB[j*N+i] = 1.f*(j-i);
	    }
	}

	// Allocate memory on the device
	int size = N*N*sizeof(float);	// Size of the memory in bytes
	float *dA,*dB,*dC;
	cudaMalloc(&dA,size);
	cudaMalloc(&dB,size);
	cudaMalloc(&dC,size);

	dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(K,K);
	
	// Copy matrices from the host to device
	cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
	cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);
	
	// Do the matrix multiplication on the GPU
	gpuClock = clock();
	addition<<<grid,threadBlock>>>(dA,dB,dC,N);
	gpuClock = clock() - gpuClock;
	
	
	// Now do the matrix multiplication on the CPU
	cpuClock = clock();
	float sum;
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			sum = 0.f;
			sum = hA[row*N+col]+hB[row*N+col];
			hC[row*N+col] = sum;
		}
	}
	cpuClock = clock() - cpuClock;
	
	// Allocate memory to store the GPU answer on the host
	float *C;
	C = new float[N*N];
	
	// Now copy the GPU result back to CPU
	cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);
	
	// Check the result and make sure it is correct
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			if( C[row*N+col] != hC[row*N+col] ){
				cout << "Wrong answer!" << endl;
				row = col = N;
			}
		}
	}
	
	printf("The CPU took %f seconds to perform matrix addition. \n", ((float)cpuClock)/CLOCKS_PER_SEC);
	printf("The GPU took %f seconds to perform matrix addition. \n", ((float)gpuClock)/CLOCKS_PER_SEC);
		
}

#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

#define BLOCK_SIZE 16

__global__ void tranposition(float *A, float *B, int N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    B[row*N+col] = A[col*N+row];
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
	float *hA,*hB;
	hA = new float[N*N];
	hB = new float[N*N];

	// Initialize matrices on the host
	for (int j=0; j<N; j++){
	    for (int i=0; i<N; i++){
	    	hA[j*N+i] = 2.f*(j+i);
	    	hA[j*N+i] = 1.f*(j+i);
	    }
	}

	// Allocate memory on the device
	int size = N*N*sizeof(float);	// Size of the memory in bytes
	float *dA,*dB;
	cudaMalloc(&dA,size);
	cudaMalloc(&dB,size);

	dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(K,K);
	
	// Copy matrices from the host to device
	cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
	
	// Do the matrix multiplication on the GPU
	gpuClock = clock();
	tranposition<<<grid,threadBlock>>>(dA,dB,N);
	gpuClock = clock() - gpuClock;
	
	
	// Now do the matrix tranposition on the CPU
	cpuClock = clock();
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
            hB[row*N+col] = hA[col*N+row];
		}
	}
	cpuClock = clock() - cpuClock;
	
	// Allocate memory to store the GPU answer on the host
	float *B;
	B = new float[N*N];
	
	// Now copy the GPU result back to CPU
	cudaMemcpy(B,dB,size,cudaMemcpyDeviceToHost);
	
	// Check the result and make sure it is correct
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			if( B[row*N+col] != hB[row*N+col] ){
				cout << "Wrong answer!" << endl;
				row = col = N;
			}
		}
	}
	
	printf("The CPU took %f seconds to perform matrix transposition. \n", ((float)cpuClock)/CLOCKS_PER_SEC);
	printf("The GPU took %f seconds to perform matrix transposition. \n", ((float)gpuClock)/CLOCKS_PER_SEC);
		
}

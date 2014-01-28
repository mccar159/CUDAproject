#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_SIZE 10
#define N (2048*2048)
#define THREADS_PER_BLOCK 512



int** makeArray(int size){
	int** extra = (int**)malloc(sizeof(int*) * size);
	for(int x=0; x<size; x++){
		extra[x] = (int*)malloc(sizeof(int) * size);
	}
	return extra;
}

void fillArray(int** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			sa[i][j] = 1;
		}
	}
}

void printArray(int** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			printf("%d ", sa[i][j]);
		}
		printf("\n");
	}
}

void freeArray(int** sa){
    for (int i = 0; i < MATRIX_SIZE; i++) {
        free(sa[i]);
    }
    free(sa);
}

__global__ void addition(int** d_a, int** d_b, int** d_c)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i =0; i < row; i++){
		for(int j =0; j < col; j++){
			d_c[i][j] = d_a[i][j] + d_b[i][j];
		}
	}
}
/*
__global__ void multiplication(int** sa1, int** sa2, int** da)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.y;
    for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			da[i][j] = 0;			
			//for(int k=0; k<row; k++){
				//da[i][j] += sa1[i][k] * sa2[k][j];
			//}
		}
	}
}

__global__ void tranpose(int** sa1, int** sa2, int** da)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.y;
	for(int i =0; i < row; i++){
		for(int j =0; j < col; j++){
			da[i][j] = sa1[j][i];
		}

	}
}
*/

int main()
{
    int **a, **b, **c;
    int **d_a, **d_b, **d_c;
     
    
    /* allocate space for host copies of a, b, c and setup input values */
    a = makeArray(MATRIX_SIZE);
    b = makeArray(MATRIX_SIZE);
    c = makeArray(MATRIX_SIZE);
       
    fillArray(a);
    fillArray(b);
    
    /* allocate space for device copies of a, b, c */
	cudaMalloc( (void **) &d_a, MATRIX_SIZE * MATRIX_SIZE * sizeof(int) );
	cudaMalloc( (void **) &d_b, MATRIX_SIZE * MATRIX_SIZE * sizeof(int) );
	cudaMalloc( (void **) &d_c, MATRIX_SIZE * MATRIX_SIZE * sizeof(int) );

	/* copy inputs to device */
	cudaMemcpy( d_a, a, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	addition <<< N/THREADS_PER_BLOCK , THREADS_PER_BLOCK >>> (d_a, d_b, d_c); 

	/* copy result back to host */

	cudaMemcpy( c, d_c, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost );

    printArray(c);

	/* clean up */

	free(a);
	free(b);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );


	
	return 0;
} /* end main */




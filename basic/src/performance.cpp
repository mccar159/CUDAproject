#include "cuda.h"
#include <stdlib.h>
#include <time.h>
#define MATRIX_SIZE 1000

int** makeArray(int size);
void fillArray(int** sa);
void printArray(int** sa);

int main( int argc, const char* argv[] )
{
	int** arr1 = makeArray(MATRIX_SIZE);
	int** arr2 = makeArray(MATRIX_SIZE);

	fillArray(arr1);
	fillArray(arr2);

	CUDA obj(MATRIX_SIZE);
	int** sum = makeArray(MATRIX_SIZE);
	int** product = makeArray(MATRIX_SIZE);
	int** transpose = makeArray(MATRIX_SIZE);

	clock_t t;
	t = clock();
	obj.addition( arr1,arr2,sum );
	t = clock() - t;
	printf("Addition, took %f seconds to complete.\n\n", ((float)t)/CLOCKS_PER_SEC);

	t = clock();
	obj.multiplication( arr1,arr2,product );
	t = clock() - t;
	printf("Multiplication, took %f seconds to complete.\n\n", ((float)t)/CLOCKS_PER_SEC);
	
	t = clock();
	obj.transpose( arr1,transpose );
	t = clock() - t;
	printf("Transpose, took %f seconds to complete.\n\n", ((float)t)/CLOCKS_PER_SEC);

	
	/*
	cout << "Inputs:" << endl;
	printArray( arr1 );
	cout << endl;
	printArray( arr2 );
	cout << endl;
	

	cout << "Outputs:" << endl;
	printArray( sum );
	cout << endl;
	printArray( product );
	cout << endl;
	printArray( transpose );
	cout << endl;
	*/
	
}

int** makeArray(int size){
	int**  extra = new int*[size];
	for(int x=0; x<size; x++){
		extra[x] = new int[size];
	}
	return extra;
}

void fillArray(int** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			sa[i][j] = rand() % 99;
		}
	}
}

void printArray(int** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			cout << sa[i][j] << " ";
		}
		cout << endl;
	}
}

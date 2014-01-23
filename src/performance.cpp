#include "cuda.h"
#include <stdlib.h>
#define MATRIX_SIZE 10

double** makeArray(int size);
void fillArray(double** sa);
void printArray(double** sa);

int main( int argc, const char* argv[] )
{
	double** arr1 = makeArray(MATRIX_SIZE);
	double** arr2 = makeArray(MATRIX_SIZE);

	fillArray(arr1);
	fillArray(arr2);

	CUDA obj(MATRIX_SIZE);
	double** sum = makeArray(MATRIX_SIZE);
	double** product = makeArray(MATRIX_SIZE);
	double** transpose = makeArray(MATRIX_SIZE);

	obj.addition( arr1,arr2,sum );
	obj.multiplication( arr1,arr2,product );
	obj.transpose( arr1,transpose );
	
	/*
	cout << "Inputs:" << endl;
	printArray( arr1 );
	cout << endl;
	printArray( arr2 );
	cout << endl;
	*/

	cout << "Outputs:" << endl;
	printArray( sum );
	cout << endl;
	printArray( product );
	cout << endl;
	printArray( transpose );
	cout << endl;
	
}

double** makeArray(int size){
	double**  extra = new double*[size];
	for(int x=0; x<size; x++){
		extra[x] = new double[size];
	}
	return extra;
}

void fillArray(double** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			sa[i][j] = rand() % 99;
		}
	}
}

void printArray(double** sa){
	for(int i=0; i<MATRIX_SIZE; i++){
		for(int j=0; j<MATRIX_SIZE; j++){
			cout << sa[i][j] << " ";
		}
		cout << endl;
	}
}

#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <iostream>
using namespace std;

class CUDA{
public:
	CUDA(int s){
		size = s;
	}
	virtual ~CUDA(){
	}
	void addition(int** sa1, int** sa2, int** da);
	void multiplication(int** sa1, int** sa2, int** da);
	void transpose(int** sa1, int** da);
private:
	int size;
};

void  CUDA::addition(int** sa1, int** sa2, int** da){
	for(int i =0; i < size; i++){
		for(int j =0; j < size; j++){
			da[i][j] = sa1[i][j] + sa2[i][j];
		}
	}
}

void CUDA::multiplication(int** sa1, int** sa2, int** da){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			da[i][j] = 0;			
			for(int k=0; k<size; k++){
				da[i][j] += sa1[i][k] * sa2[k][j];
			}
		}
	}
}

void CUDA::transpose(int** sa1, int** da){
	for(int i =0; i < size; i++){
		for(int j =0; j < size; j++){
			da[i][j] = sa1[j][i];
		}

	}
}

#endif

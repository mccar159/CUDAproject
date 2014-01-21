#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
using namespace std;

class CUDA{
public:
	CUDA(){
		size =0;
	}
	double** addition(double** arr1, double** arr2);
	void multiplication(double** arr1, double** arr2);
	double** transpose(double** arr1);
private:
	int size;

};

double** CUDA::addition(double** arr1, double** arr2){
	for(int i =0; i < arr1.size; i++){
		for(int j =0; j < arr1.size; j++){
			sum[i][j] = arr1[i][j] + arr2[i][j];
		}

	}
}

void CUDA::multiplication(double** arr1, double** arr2){
	for(int i = 0; i < size; i++){
		double buff = 0;
		for(int j =0; j < size, j++){
			for(int k =0; k < arr1.size(); k++){
				buff += arr1[i][k] * arr2[k][j];
			}
			cout << buff << endl;
		}
	}
}

double** CUDA::transpose(double** arr1){
	for(int i =0; i < size; i++){
		for(int j =0; j < size; j++){
			final[i][j] = arr1[j][i];
		}

	}
}

#endif

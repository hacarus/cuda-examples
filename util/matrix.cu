#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>

void showMatrix(float *x, int m, int n, bool gpu = 1){
    float *y;
    if(gpu){
        y = (float*)calloc(m*n, sizeof(float));
        cudaMemcpy(y, x, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    }else{
        y = x;
    }
    for(int i = 0;i < m;++i){
        for(int j = 0; j< n;++j){
            printf("%f ", y[i*n+j]);
        }
        printf("\n");
    }
    if(gpu){
        free(y);
    }
}
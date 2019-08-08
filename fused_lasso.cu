#include <iostream>
#include <random>
#include "cuda/lasso_admm.cu"
#include "util/matrix.cu"
#include "util/load_matrix.c"
#include "util/perf_counter.cpp"

int main(){
    float *X;
    float *y;
    int mX, nX, my, ny;
    loadMatrix("data/fused/X.txt", &X, &mX, &nX);
    loadMatrix("data/fused/y.txt", &y, &my, &ny);
    std::cout<<"mX:"<<mX<<" nX:"<<nX<<std::endl;
    std::cout<<"my:"<<my<<" ny:"<<ny<<std::endl;
    fused_D D;
    D.sparse_coef = 0;
    D.trend_coef = 0.05;
    int n_iter[1];
    perf_counter ctr;
	
    float *X_;
    float *y_;
    float *w_;
    cudaMalloc(&X_, mX*nX*sizeof(float));
    cudaMalloc(&y_, my*ny*sizeof(float));
    cudaMalloc(&w_, my*ny*sizeof(float));
    cudaMemcpy(X_, X, mX*nX*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, my*ny*sizeof(float), cudaMemcpyHostToDevice);
    ctr.start();
    int a = _admm(X_, y_, D, mX, nX, ny, 0.04, 0.04, 1e-7, 5000 ,w_, n_iter);
    ctr.stop();
    std::cout<< "Exec time:\t"	<< ctr << "\tmsec" << std::endl;
    std::cout<< "N iter:\t" << n_iter[0]<<std::endl;
    float *w = (float*)calloc(my*ny, sizeof(float));
    cudaMemcpy(w, w_, my*ny*sizeof(float), cudaMemcpyDeviceToHost);
    saveMatrix("w.txt", w, my, ny);
    
    cudaFree(X_);
    cudaFree(y_);
    cudaFree(w_);

    free(X);
    free(y);
    free(w);
    return 0;
}


#include <iostream>
#include <random>
#include "cuda/lasso_admm.cu"
#include "util/load_matrix.c"
#include "util/perf_counter.cpp"

int main(){
    float *X;
    float *y;
    int mX, nX, my, ny;
    loadMatrix("X.txt", &X, &mX, &nX);
    loadMatrix("y.txt", &y, &my, &ny);
    std::cout<<"mX:"<<mX<<" nX:"<<nX<<std::endl;
    std::cout<<"my:"<<my<<" ny:"<<ny<<std::endl;
    transform_matrix D;
    float *X_;
    float *y_;
    float *w_;
    cudaMalloc(&w_, my*ny*sizeof(float));
    cudaMalloc(&X_, mX*nX*sizeof(float));
    cudaMalloc(&y_, my*ny*sizeof(float));
    cudaMemcpy(X_, X, mX*nX*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, my*ny*sizeof(float), cudaMemcpyHostToDevice);
    int n_iter[1];
    perf_counter ctr;
    ctr.start();
    _admm(X_, y_, D, mX, nX, ny, 0.3, 1., 1e-8, 100,  w_, n_iter);
	ctr.stop();
    std::cout<< "Exec time:"	<< ctr << "msec" << std::endl;
    std::cout<< "N iter:" << n_iter[0]<<std::endl;
    showMatrix(w_, 1, 13);
    cudaFree(X_);
    cudaFree(y_);
    cudaFree(w_);

    free(X);
    free(y);
    return 0;
}


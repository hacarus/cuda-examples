#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "transform_matrix.cu"
#include "cuda_inv.cu"

const float ZERO = 0;
const float ONE = 1;
const float MINUS_ONE = -1;


__device__ float _soft_threshold(float x, float thresh){
    if(x>thresh){
        return x - thresh;
    }else if(x < -thresh){
        return x + thresh;
    }else{
        return 0;
    }
}

__global__ void
_uzh(float *Dw_t, float *h_k, float *z_k, float *sub_z_h_k, float threshold, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=n) return;
    float dw = Dw_t[index];
    float h = h_k[index];
    float z = _soft_threshold(dw + h, threshold);
    z_k[index] = z;
    h += dw - z;
    h_k[index] = h;
    sub_z_h_k[index] = z - h;
}

void _update_z_h(float *Dw_t, float *h_k, float *z_k, float *sub_z_h_k, float threshold, int n)
{
    _uzh<<<n / 32 + (n % 32 ? 1 : 0), 32>>>(Dw_t, h_k, z_k, sub_z_h_k, threshold, n);
}

float _cost_function(float *X, float *y, float *w, float *z, float alpha, int n_samples, int n_features, cublasHandle_t cublas_handle){
    float loss_w, loss_z;
    cublasSgemv(cublas_handle, CUBLAS_OP_T,
        n_features, n_samples,
                &ONE,
                X, n_features,
                w, 1,
                &MINUS_ONE,
                y,
                1);
    
    cublasSnrm2(cublas_handle, n_samples, y, 1, &loss_w);
    cublasSasum(cublas_handle, n_features, z, 1, &loss_z);
    cudaDeviceSynchronize();
    return loss_w / n_samples + alpha * loss_z;
}

int
_update(float *X, float *y_k, transform_matrix &D,
        int n_samples, int n_features, int n_targets, int n,
        float *coef_matrix, float *inv_Xy_k, float *inv_D,
        float alpha, float rho, int max_iter, float tol,
        cublasHandle_t cublas_handle, float *w, int *n_iter)
{
    // n_samples, n_features = X.shape
    // n_samples, 1 = y_k
    // n_features, 1 = w_k.shape
    // n_features, 1 = inv_Xy_k.shape
    
    float inv_n_samples = 1.0f / n_samples;
    float threshold = alpha / rho;
    float *w_k;
    float *Dw_t;
    float *z_k;
    float *h_k;
    float *sub_z_h_k;
    float *cost_workspace_y;
    int status = 0;
    status+=cudaMalloc(&cost_workspace_y, n_samples * sizeof(float));
    status+=cudaMalloc(&w_k, n_features * sizeof(float));
    status+=cudaMalloc(&Dw_t, n_features * sizeof(float));
    status+=cudaMalloc(&z_k, n_features * sizeof(float));
    status+=cudaMalloc(&h_k, n_features * sizeof(float));
    status+=cudaMalloc(&sub_z_h_k, n_features * sizeof(float));

    status+=cublasSgemv(cublas_handle, CUBLAS_OP_N,
        n_features, n_samples,
                &inv_n_samples,
                X, n_features,
                y_k, 1,
                &ZERO,
                w_k,
                1);
    status+=cudaDeviceSynchronize();
    D(w_k, z_k, n_features);
    status+=cudaMemset(h_k, 0, n_features * sizeof(float));
    status+=cudaMemcpy(sub_z_h_k, z_k, n_features * sizeof(float), cudaMemcpyDeviceToDevice);
    status+=cudaDeviceSynchronize();
    status+=cudaMemcpy(cost_workspace_y, y_k, n_samples * sizeof(float), cudaMemcpyDeviceToDevice);
    float cost = _cost_function(X, cost_workspace_y, w_k, z_k, alpha, n_samples, n_features, cublas_handle);
    float pre_cost, gap;
    int i;
    for (i = 0; i < max_iter; ++i){
        status+=cudaMemcpy(w_k, inv_Xy_k, n_features * sizeof(float), cudaMemcpyDeviceToDevice);
        
        status+=cublasSgemv(cublas_handle, CUBLAS_OP_T,
            n_features, n_features,
                    &ONE,
                    inv_D, n_features,
                    sub_z_h_k, 1,
                    &ONE,
                    w_k,
                    1);
        status+=cudaDeviceSynchronize();

        D(w_k, Dw_t, n_features);

        status+=cudaDeviceSynchronize();
        _update_z_h(Dw_t, h_k, z_k, sub_z_h_k, threshold, n_features);
        status+=cudaDeviceSynchronize();
        pre_cost = cost;
        status+=cudaMemcpy(cost_workspace_y, y_k, n_samples * sizeof(float), cudaMemcpyDeviceToDevice);
        cost = _cost_function(X, cost_workspace_y, w_k, z_k, alpha, n_samples, n_features, cublas_handle);
        gap = abs(cost - pre_cost);
        if(gap < tol){
            break;
        }
    }
    n_iter[n] = i;

    status+=cudaMemcpy(w + n_features * n, w_k, n_features*sizeof(float), cudaMemcpyDeviceToDevice);

    status+=cudaFree(cost_workspace_y);
    status+=cudaFree(w_k);
    status+=cudaFree(Dw_t);
    status+=cudaFree(z_k);
    status+=cudaFree(h_k);
    status+=cudaFree(sub_z_h_k);
    return status;
}

int _admm(float *X, float *y, transform_matrix &D,
    int n_samples, int n_features, int n_targets,
    float alpha, float rho, float tol, int max_iter, float *w, int* n_iter)
{
    // n_samples, n_features = X.shape
    // n_features, n_features = coef_matrix.shape == inv_matrix.shape
    // n_features, n_samples = inv_matrix_dot_X_T.shape
    // n_samples, n_targets = y.shape
    // n_features, n_targets = inv_Xy.shape

    int status=0;
    cublasHandle_t cublas_handle;
    status+=cublasCreate(&cublas_handle);
    float *w_t;
    status+=cudaMalloc(&w_t, n_features*n_targets*sizeof(float));
    float *coef_matrix;

    status+=cudaMalloc(&coef_matrix, n_features * n_features * sizeof(float));
    D.DTD(coef_matrix, n_features);
    status+=cudaDeviceSynchronize();
    float inv_n_samples = 1.0f / n_samples;
    status+=cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                n_features, n_features, n_samples,
                &inv_n_samples,
                X, n_features,
                X, n_features,
                &rho,
                coef_matrix,
                n_features);

    float *inv_matrix;
    status+=cudaMalloc(&inv_matrix, sizeof(float) * n_features * n_features);
    status+=cudaDeviceSynchronize();
    status+=inv(coef_matrix, inv_matrix, n_features);
    status+=cudaDeviceSynchronize();
    float *inv_Xy, *inv_matrix_dot_X_T;
    status+=cudaMalloc(&inv_matrix_dot_X_T, n_features * n_samples * sizeof(float));
    status+=cudaMalloc(&inv_Xy, n_features * n_targets * sizeof(float));


    status+=cudaDeviceSynchronize();
    status+=cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n_samples, n_features, n_features,
                &ONE,
                X, n_features,
                inv_matrix, n_features,
                &ZERO,
                inv_matrix_dot_X_T,
                n_samples);
    status+=cudaDeviceSynchronize();

    status+=cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                n_features, n_targets, n_samples,
                &inv_n_samples,
                inv_matrix_dot_X_T, n_samples,
                y, n_targets,
                &ZERO,
                inv_Xy,
                n_features);
    
                
    float *inv_D;
    status+=cudaMalloc(&inv_D, n_features * n_features * sizeof(float));
    D.D_B(inv_matrix, inv_D, n_features, rho);
    status+=cudaDeviceSynchronize();

    for (int i = 0; i < n_targets;++i){
        status+=_update(X, y, D,
                n_samples, n_features, n_targets, i,
                coef_matrix, inv_Xy + n_features * i, inv_D,
                alpha, rho, max_iter, tol,
                cublas_handle, w, n_iter);
    }
    status+=cudaDeviceSynchronize();

    status+=cudaFree(w_t);
    status+=cudaFree(coef_matrix);
    status+=cudaFree(inv_matrix);
    status+=cudaFree(inv_matrix_dot_X_T);
    status+=cudaFree(inv_Xy);
    status+=cudaFree(inv_D);
    status+=cublasDestroy(cublas_handle);
    return status;
}


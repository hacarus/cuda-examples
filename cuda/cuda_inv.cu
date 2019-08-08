#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define MSG(a) std::cout << #a << " " << a << std::endl;

int inv(const float *A, float* B, int n)
{
    cusolverStatus_t status, status0, status1, status2;
    cusolverDnHandle_t handle;
    status = cusolverDnCreate(&handle);
    
    mkide(B, n);
    float *A_;
    cudaMalloc(&A_, sizeof(float) * n * n);
    cudaMemcpy(A_, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
    int worksize;
    float *workspace;
    int *devInfo;
    int *devIpiv;
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&devIpiv, sizeof(int)*n);

    status0 = cusolverDnSgetrf_bufferSize(handle,
        n, n,
        A_,
        n,
        &worksize);
    cudaMalloc(&workspace, sizeof(float)*worksize);
    
    status1 = cusolverDnSgetrf(handle,
        n, n,
        A_,
        n,
        workspace,
        devIpiv,
        devInfo);

    status2 = cusolverDnSgetrs(handle,
        CUBLAS_OP_N,
        n,
        n,
        A_,
        n,
        devIpiv,
        B,
        n,
        devInfo);
        cusolverDnDestroy(handle);
    return status+status0+status1+status2;
}

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

typedef union{
    float f;
    int i;
}f2i;

inline void mkide(float *a, int n)
{
    f2i d;
    d.f = 1.0f;
    cudaMemset(a, 0, n*n*sizeof(float));
    cuMemsetD2D32((CUdeviceptr)a, (n+1)*sizeof(float), d.i, 1, n);
}

__global__ void D_B(float *x, float *y, int n, float rho)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)return;
    y[index] = x[index]*rho;
}

typedef struct
{
    virtual void operator()(float *x, float *y, int n)
    {
        cudaMemcpy(y, x, n * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    virtual void DTD(float *x, int n)
    {
        mkide(x, n);
    }
    virtual void D_B(float *x, float *y, int n)
    {
        cudaMemcpy(y, x, n * n * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    virtual void D_B(float *x, float *y, int n, float rho)
    {
        D_B(x, y, n*n, rho);
    }

} transform_matrix;

__global__ void fused_D_N(float *x, float *y, int n, float sparse_coef, float trend_coef, float diag_coef)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)return;
    float t0, t1 = x[index];
    y[index] = t1 * sparse_coef;
    for (int i = 1; i < n; ++i)
    {
        t0 = t1;
        t1 = x[n * i + index];
        y[n * i + index] = t1 * diag_coef - t0 * trend_coef;
    }
}

__global__ void fused_D_N_1d(float *x, float *y, int n, float sparse_coef, float trend_coef, float diag_coef)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)return;
    
    float t0 = x[index];
    float u = __shfl(t0, index-1);
    if(index==0){
        y[index]=sparse_coef*t0;
    }else if(threadIdx.x==0){
        y[index]= t0*diag_coef-x[index-1]*trend_coef;
    }else{
        y[index]= t0*diag_coef-u*trend_coef;
    }

}


__global__ void fused_D_B(float *x, float *y, int n, float sparse_coef, float trend_coef, float diag_coef)
{
    int j = threadIdx.x;
    int j_ = blockDim.x;
    int i = blockIdx.x;
    int index = i * n + j;
    int target = (j - 1 + j_) % j_;
    float t = x[index];
    float u0 = 0;
    float u1 = __shfl(t, target, j_);
    float &u = j == 0 ? u0 : u1;
    
    y[index] = t * (j==0?sparse_coef:diag_coef) - u * trend_coef;

    for (int i = 1; i < n/32; ++i)
    {
        index += j_;
        u0 = u1;
        t = x[index];
        u1 = __shfl(t, target, j_);
        y[index] = t * diag_coef - u * trend_coef;
    }
    index += j_;
    if(index/n!=i)
    {
        __shfl(0, target, 0);
    }else
    {
        t = x[index];
	    u0 = u1;
        u1 = __shfl(t, target, j_);
        y[index] = t * diag_coef - u * trend_coef;
    }
}

__global__ void fused_DTD(float *x, int n, float t0, float t1, float t2, float t3)
{
    int i = threadIdx.x % 3 - 1;
    int j = threadIdx.x / 3 + blockIdx.x * 10;
    int index = j * n + j + i;
    float t;
    if (index >= n * n || index < 0)
        return;
    if (index == 0)
        t = t0;
    else if (index == n * n - 1)
        t = t3;
    else if (i == 0)
        t = t2;
    else
        t = t1;

    x[index] = t;
}

struct fused_D :public transform_matrix
{
    float sparse_coef, trend_coef;
    virtual void operator()(float *x, float *y, int n)
    {
        float diag_coef = sparse_coef + trend_coef;
        fused_D_N_1d<<<n / 32 + (n % 32 ? 1 : 0), 32>>>(x, y, n, sparse_coef, trend_coef, diag_coef);
        // fused_D_N<<<n / 32 + (n % 32 ? 1 : 0), 32>>>(x, y, n, sparse_coef, trend_coef, diag_coef);
    }
    virtual void DTD(float *x, int n)
    {
	float spt = sparse_coef + trend_coef;
        float t0 = sparse_coef * sparse_coef + trend_coef * trend_coef;
        float t1 = -(spt) * trend_coef;
        float t2 = spt*spt + trend_coef * trend_coef;
        float t3 = spt * spt;
        fused_DTD<<<n / 10 + (n % 10 ? 1 : 0), 30>>>(x, n, t0, t1, t2, t3);
    }
    virtual void D_B(float *x, float *y, int n)
    {
        float diag_coef = sparse_coef + trend_coef;
        fused_D_B<<<n, 32>>>(x, y, n, sparse_coef, trend_coef, diag_coef);
    }

   virtual void D_B(float *x, float *y, int n, float rho)
   {
       float diag_coef = sparse_coef + trend_coef;
       fused_D_B<<<n, 32>>>(x, y, n, sparse_coef * rho, trend_coef * rho, diag_coef * rho);
   }

};
typedef struct fused_D fused_D;

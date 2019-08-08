#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

void loadMatrix(char *filepass, float **x, int *m, int *n)
{
    FILE *fp = fopen(filepass, "r");
    fscanf(fp, "%d %d", m, n);
    *x = (float*)calloc(*m**n, sizeof(float));
    for (int i = 0; i < *m**n; ++i)
    {
        fscanf(fp, "%f", &(*x)[i]);
    }
    fclose(fp);
}

void saveMatrix(char *filepass, float *x, int m, int n)
{
    FILE *fp = fopen(filepass, "w");
    fprintf(fp, "%d %d\n", m, n);
    for (int i = 0; i < m; ++i)
    {
        for(int j = 0;j<n;++j){
            fprintf(fp, "%f ", x[i*n+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

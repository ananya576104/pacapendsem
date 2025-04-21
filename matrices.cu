#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_LEN 1024

// ============= CUDA KERNELS ==================== //

__global__ void matrixAdd(int* A, int* B, int* C, int m, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matrixMultiply(int* A, int* B, int* C, int m, int n, int q) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < q) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * q + col];
        }
        C[row * q + col] = sum;
    }
}

__global__ void matrixTranspose(int* A, int* B, int m, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < n) {
        B[col * m + row] = A[row * n + col];
    }
}

__global__ void repeatChars(char* A, int* B, char* STR, int* offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int rep = B[idx];
        int start = offset[idx];
        for (int i = 0; i < rep; i++) {
            STR[start + i] = A[idx];
        }
    }
}

__global__ void rowSquareCube(int* A, int* sq, int* cube, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        sq[idx] = A[idx] * A[idx];
        cube[idx] = A[idx] * A[idx] * A[idx];
    }
}

__global__ void rowSumProduct(int* A, int* rowSum, int* rowProd, int m, int n) {
    int row = blockIdx.x;
    if (row < m) {
        int sum = 0, prod = 1;
        for (int j = 0; j < n; j++) {
            int val = A[row * n + j];
            sum += val;
            prod *= val;
        }
        rowSum[row] = sum;
        rowProd[row] = prod;
    }
}

__global__ void colSumProduct(int* A, int* colSum, int* colProd, int m, int n) {
    int col = blockIdx.x;
    if (col < n) {
        int sum = 0, prod = 1;
        for (int i = 0; i < m; i++) {
            int val = A[i * n + col];
            sum += val;
            prod *= val;
        }
        colSum[col] = sum;
        colProd[col] = prod;
    }
}

// ============= HOST FUNCTIONS ================= //

void computeOffset(int* B, int* offset, int size) {
    offset[0] = 0;
    for (int i = 1; i < size; i++) {
        offset[i] = offset[i - 1] + B[i - 1];
    }
}

void printMatrix(int* mat, int rows, int cols, const char* name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows * cols; i++) {
        printf("%d ", mat[i]);
        if ((i + 1) % cols == 0) printf("\n");
    }
}

int main() {
    int m, n, q;

    printf("Enter dimensions of Matrix A (m x n): ");
    scanf("%d %d", &m, &n);
    printf("Enter number of columns for Matrix B (n x q): ");
    scanf("%d", &q);

    int sizeA = m * n;
    int sizeB = n * q;

    int* h_A = (int*)malloc(sizeA * sizeof(int));
    int* h_B = (int*)malloc(sizeB * sizeof(int));
    char* h_CHA = (char*)malloc(sizeA * sizeof(char));
    int* h_REP = (int*)malloc(sizeA * sizeof(int));

    printf("Enter elements of Matrix A (%d elements):\n", sizeA);
    for (int i = 0; i < sizeA; i++) scanf("%d", &h_A[i]);

    printf("Enter elements of Matrix B (%d elements):\n", sizeB);
    for (int i = 0; i < sizeB; i++) scanf("%d", &h_B[i]);

    printf("Enter %d characters for expansion:\n", sizeA);
    for (int i = 0; i < sizeA; i++) scanf(" %c", &h_CHA[i]);

    printf("Enter repetition counts for each character (%d values):\n", sizeA);
    for (int i = 0; i < sizeA; i++) scanf("%d", &h_REP[i]);

    int* h_Add = (int*)malloc(sizeA * sizeof(int));
    int* h_Mul = (int*)malloc(m * q * sizeof(int));
    int* h_Trans = (int*)malloc(n * m * sizeof(int));
    int* h_Sq = (int*)malloc(sizeA * sizeof(int));
    int* h_Cube = (int*)malloc(sizeA * sizeof(int));
    int* h_RowSum = (int*)malloc(m * sizeof(int));
    int* h_RowProd = (int*)malloc(m * sizeof(int));
    int* h_ColSum = (int*)malloc(n * sizeof(int));
    int* h_ColProd = (int*)malloc(n * sizeof(int));
    int* h_offset = (int*)malloc(sizeA * sizeof(int));

    computeOffset(h_REP, h_offset, sizeA);
    int totalLen = h_offset[sizeA - 1] + h_REP[sizeA - 1];
    char* h_STR = (char*)malloc((totalLen + 1) * sizeof(char));

    // Device memory
    int *d_A, *d_B, *d_Add, *d_Mul, *d_Trans, *d_Sq, *d_Cube;
    int *d_RowSum, *d_RowProd, *d_ColSum, *d_ColProd;
    char *d_CHA, *d_STR;
    int *d_REP, *d_offset;

    cudaMalloc(&d_A, sizeA * sizeof(int));
    cudaMalloc(&d_B, sizeB * sizeof(int));
    cudaMalloc(&d_Add, sizeA * sizeof(int));
    cudaMalloc(&d_Mul, m * q * sizeof(int));
    cudaMalloc(&d_Trans, n * m * sizeof(int));
    cudaMalloc(&d_Sq, sizeA * sizeof(int));
    cudaMalloc(&d_Cube, sizeA * sizeof(int));
    cudaMalloc(&d_RowSum, m * sizeof(int));
    cudaMalloc(&d_RowProd, m * sizeof(int));
    cudaMalloc(&d_ColSum, n * sizeof(int));
    cudaMalloc(&d_ColProd, n * sizeof(int));
    cudaMalloc(&d_CHA, sizeA * sizeof(char));
    cudaMalloc(&d_STR, totalLen * sizeof(char));
    cudaMalloc(&d_REP, sizeA * sizeof(int));
    cudaMalloc(&d_offset, sizeA * sizeof(int));

    // Copy to device
    cudaMemcpy(d_A, h_A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CHA, h_CHA, sizeA * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_REP, h_REP, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset, sizeA * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 gridAdd((n + 15) / 16, (m + 15) / 16);
    dim3 gridMul((q + 15) / 16, (m + 15) / 16);

    matrixAdd<<<gridAdd, block>>>(d_A, d_A, d_Add, m, n);
    matrixMultiply<<<gridMul, block>>>(d_A, d_B, d_Mul, m, n, q);
    matrixTranspose<<<gridAdd, block>>>(d_A, d_Trans, m, n);
    rowSquareCube<<<(sizeA + 255) / 256, 256>>>(d_A, d_Sq, d_Cube, sizeA);
    rowSumProduct<<<m, 1>>>(d_A, d_RowSum, d_RowProd, m, n);
    colSumProduct<<<n, 1>>>(d_A, d_ColSum, d_ColProd, m, n);
    repeatChars<<<(sizeA + 255) / 256, 256>>>(d_CHA, d_REP, d_STR, d_offset, sizeA);

    cudaMemcpy(h_Add, d_Add, sizeA * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Mul, d_Mul, m * q * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Trans, d_Trans, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Sq, d_Sq, sizeA * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cube, d_Cube, sizeA * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_RowSum, d_RowSum, m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_RowProd, d_RowProd, m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ColSum, d_ColSum, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ColProd, d_ColProd, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_STR, d_STR, totalLen * sizeof(char), cudaMemcpyDeviceToHost);
    h_STR[totalLen] = '\0';

    // Output
    printMatrix(h_Add, m, n, "Matrix Addition (A + A)");
    printMatrix(h_Mul, m, q, "Matrix Multiplication (A x B)");
    printMatrix(h_Trans, n, m, "Transpose of A");
    printMatrix(h_Sq, m, n, "Element-wise Square");
    printMatrix(h_Cube, m, n, "Element-wise Cube");

    printf("\nRow Sum:\n");
    for (int i = 0; i < m; i++) printf("%d ", h_RowSum[i]);
    printf("\nRow Product:\n");
    for (int i = 0; i < m; i++) printf("%d ", h_RowProd[i]);

    printf("\nColumn Sum:\n");
    for (int i = 0; i < n; i++) printf("%d ", h_ColSum[i]);
    printf("\nColumn Product:\n");
    for (int i = 0; i < n; i++) printf("%d ", h_ColProd[i]);

    printf("\n\nExpanded String:\n%s\n", h_STR);

    // Free
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Add); cudaFree(d_Mul);
    cudaFree(d_Trans); cudaFree(d_Sq); cudaFree(d_Cube);
    cudaFree(d_RowSum); cudaFree(d_RowProd); cudaFree(d_ColSum); cudaFree(d_ColProd);
    cudaFree(d_CHA); cudaFree(d_STR); cudaFree(d_REP); cudaFree(d_offset);
    free(h_A); free(h_B); free(h_CHA); free(h_REP); free(h_Add); free(h_Mul);
    free(h_Trans); free(h_Sq); free(h_Cube); free(h_RowSum); free(h_RowProd);
    free(h_ColSum); free(h_ColProd); free(h_offset); free(h_STR);

    return 0;
}

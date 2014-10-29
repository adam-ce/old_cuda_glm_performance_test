/*
 *  Copyright (C) 2014 Adam Celarek
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
*/

#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "helper_math.h"
#include <stdio.h>
#include <chrono>
#include <glm/gtx/matrix_major_storage.hpp>

#define NUM_ELEMENTS 2000000
#define THREADS_PER_BLOCK 256

//handle cuda errors
void hce(cudaError_t error)
{
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

__global__ void cuMatrixKernel(const float4 *vectors, mat4 matrix, float4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = matrix * vectors[i];
        if(i > 3) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i] = matrix * result[i];
                result[i] += matrix * vectors[i-1];
                result[i] += matrix * vectors[i-2];
                result[i] += matrix * vectors[i-3];
                result[i] += matrix * vectors[i-4];
            }
        }
    }
}
__global__ void cuDotKernel(const float4 *vectors, float4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i].y = dot(vectors[i+1], vectors[i]);
                result[i].x = dot(vectors[i-1], vectors[i]);
                result[i].z = dot(vectors[i+1], vectors[0]);
                result[i].w = dot(vectors[i-1], vectors[0]);
            }
        }
    }
}

__global__ void cuCrossKernel(const float4 *vectors, float4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                float3 tmp1 = make_float3(vectors[i-1]);
                float3 tmp2 = make_float3(vectors[i]);
                float3 tmp3 = cross(tmp1, tmp2);
                result[i] = make_float4(tmp3, tmp3.x);
            }
        }
    }
}


__global__ void glmMatrixKernel(const glm::vec4 *vectors, glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = matrix * vectors[i];
        if(i > 3) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i] = matrix * result[i];
                result[i] += matrix * vectors[i-1];
                result[i] += matrix * vectors[i-2];
                result[i] += matrix * vectors[i-3];
                result[i] += matrix * vectors[i-4];
            }
        }
    }
}
__global__ void glmDotKernel(const glm::vec4 *vectors, glm::vec4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i].y = glm::dot(vectors[i+1], vectors[i]);
                result[i].x = glm::dot(vectors[i-1], vectors[i]);
                result[i].z = glm::dot(vectors[i+1], vectors[0]);
                result[i].w = glm::dot(vectors[i-1], vectors[0]);
            }
        }
    }
}

__global__ void glmCrossKernel(const glm::vec4 *vectors, glm::vec4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                glm::vec3 tmp1 = glm::vec3(vectors[i-1]);
                glm::vec3 tmp2 = glm::vec3(vectors[i]);
                glm::vec3 tmp3 = glm::cross(tmp1, tmp2);
                result[i] = glm::vec4(tmp3, tmp3.x);
            }
        }
    }
}

void cpuGlmMatrixKernel(const glm::vec4 *vectors, glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize) {
    for(int i=0; i<numElements; i++) {
        result[i] = matrix * vectors[i];
        if(i > 3) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i] = matrix * result[i];
                result[i] += matrix * vectors[i-1];
                result[i] += matrix * vectors[i-2];
                result[i] += matrix * vectors[i-3];
                result[i] += matrix * vectors[i-4];
            }
        }
    }
}
void cpuGlmDotKernel(const glm::vec4 *vectors, glm::vec4 *result, int numElements, int innerLoopSize) {
    for(int i=0; i<numElements; i++) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                result[i].y = glm::dot(vectors[i+1], vectors[i]);
                result[i].x = glm::dot(vectors[i-1], vectors[i]);
                result[i].z = glm::dot(vectors[i+1], vectors[0]);
                result[i].w = glm::dot(vectors[i-1], vectors[0]);
            }
        }
    }
}

void cpuGlmCrossKernel(const glm::vec4 *vectors, glm::vec4 *result, int numElements, int innerLoopSize) {
    for(int i=0; i<numElements; i++) {
        result[i] = vectors[i];
        if(i>1 && i < NUM_ELEMENTS - 1) {
            for(int j=0; j<innerLoopSize; j++) {
                glm::vec3 tmp1 = glm::vec3(vectors[i-1]);
                glm::vec3 tmp2 = glm::vec3(vectors[i]);
                glm::vec3 tmp3 = glm::cross(tmp1, tmp2);
                result[i] = glm::vec4(tmp3, tmp3.x);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    glm::mat4 glmMatrix;
    glmMatrix[0] = glm::vec4(1.085f, -.15f, .72f, -0.65f);
    glmMatrix[1] = glm::vec4(.35f, -.89f, .79f, -.32f);
    glmMatrix[2] = glm::vec4(.38f, -.46f, .26f, -.83f);
    glmMatrix[3] = glm::vec4(.38f, -.80f, .90f, -.50f);

    mat4 cuMatrix = make_mat4(glmMatrix);

    glm::vec4* glmVectors = new glm::vec4[NUM_ELEMENTS];
    float4* cuVectors = new float4[NUM_ELEMENTS];

    for(int i=0; i<NUM_ELEMENTS; i++) {
        glmVectors[i] = glm::vec4(rand() / (float) RAND_MAX, rand() / (float) RAND_MAX, rand() / (float) RAND_MAX, rand() / (float) RAND_MAX);
        cuVectors[i] = make_float4(glmVectors[i]);
    }
    glmVectors[0] = glm::vec4(1.f, 0.f, 1.f, 0.f);
    glmVectors[1] = glm::vec4(0.f, 1.f, 0.f, 1.f);
    cuVectors[0] = make_float4(glmVectors[0]);
    cuVectors[1] = make_float4(glmVectors[1]);

    glm::vec4* cpuResult = new glm::vec4[NUM_ELEMENTS];

    size_t glmSize = NUM_ELEMENTS * sizeof(glm::vec4);
    glm::vec4* d_glmVectors;
    hce(cudaMalloc(&d_glmVectors, glmSize));
    hce(cudaMemcpy(d_glmVectors, glmVectors, NUM_ELEMENTS * sizeof(glm::vec4), cudaMemcpyHostToDevice));
    glm::vec4* d_glmResult;
    hce(cudaMalloc(&d_glmResult, glmSize));

    size_t cuSize = NUM_ELEMENTS * sizeof(float4);
    float4* d_cuVectors;
    hce(cudaMalloc(&d_cuVectors, cuSize));
    hce(cudaMemcpy(d_cuVectors, cuVectors, NUM_ELEMENTS * sizeof(float4), cudaMemcpyHostToDevice));
    float4* d_cuResult;
    hce(cudaMalloc(&d_cuResult, cuSize));

    int blocksPerGrid = (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, THREADS_PER_BLOCK);

    //warmup
    glmMatrixKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 10);  hce(cudaDeviceSynchronize());
    glmDotKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, d_glmResult, NUM_ELEMENTS, 10);                hce(cudaDeviceSynchronize());
    glmCrossKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, d_glmResult, NUM_ELEMENTS, 10);              hce(cudaDeviceSynchronize());

    cuMatrixKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 10);      hce(cudaDeviceSynchronize());
    cuDotKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, d_cuResult, NUM_ELEMENTS, 10);                   hce(cudaDeviceSynchronize());
    cuCrossKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, d_cuResult, NUM_ELEMENTS, 10);                 hce(cudaDeviceSynchronize());

    cpuGlmMatrixKernel(glmVectors, glmMatrix, cpuResult, NUM_ELEMENTS, 10);
    cpuGlmDotKernel(glmVectors, cpuResult, NUM_ELEMENTS, 10);
    cpuGlmCrossKernel(glmVectors, cpuResult, NUM_ELEMENTS, 10);

    auto timeMatrix0 = std::chrono::high_resolution_clock::now();

    glmMatrixKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 100); hce(cudaDeviceSynchronize());
    auto timeMatrix1 = std::chrono::high_resolution_clock::now();

    cuMatrixKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 100);     hce(cudaDeviceSynchronize());
    auto timeMatrix2 = std::chrono::high_resolution_clock::now();

    cpuGlmMatrixKernel(glmVectors, glmMatrix, cpuResult, NUM_ELEMENTS, 100);
    auto timeMatrix3 = std::chrono::high_resolution_clock::now();

    glm::vec4* glmResult = new glm::vec4[NUM_ELEMENTS];
    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    float4* cuResult = new float4[NUM_ELEMENTS];
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    hce(cudaGetLastError());
    for(int i=0; i<NUM_ELEMENTS; i++) {
        if(length(cuResult[i] - make_float4(glmResult[i])) > 0.01f) {
            std::cerr << "error matrix i=" << i << std::endl;
            break;
        }
        if(glm::length(cpuResult[i] - glmResult[i]) > 0.01f) {
            std::cerr << "error matrix i=" << i << std::endl;
            break;
        }
    }

    auto timeDot0 = std::chrono::high_resolution_clock::now();

    glmDotKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, d_glmResult, NUM_ELEMENTS, 100);       hce(cudaDeviceSynchronize());
    auto timeDot1 = std::chrono::high_resolution_clock::now();

    cuDotKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, d_cuResult, NUM_ELEMENTS, 100);          hce(cudaDeviceSynchronize());
    auto timeDot2 = std::chrono::high_resolution_clock::now();

    cpuGlmDotKernel(glmVectors, cpuResult, NUM_ELEMENTS, 100);
    auto timeDot3 = std::chrono::high_resolution_clock::now();

    hce(cudaGetLastError());

    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    for(int i=0; i<NUM_ELEMENTS; i++) {
        if(length(cuResult[i] - make_float4(glmResult[i])) > 0.0001f) {
            std::cerr << "error dot i=" << i << std::endl;
            break;
        }
        if(glm::length(cpuResult[i] - glmResult[i]) > 0.01f) {
            std::cerr << "error dot i=" << i << std::endl;
            break;
        }
    }


    auto timeCross0 = std::chrono::high_resolution_clock::now();

    glmCrossKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, d_glmResult, NUM_ELEMENTS, 100);     hce(cudaDeviceSynchronize());
    auto timeCross1 = std::chrono::high_resolution_clock::now();

    cuCrossKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, d_cuResult, NUM_ELEMENTS, 100);        hce(cudaDeviceSynchronize());
    auto timeCross2 = std::chrono::high_resolution_clock::now();

    cpuGlmCrossKernel(glmVectors, cpuResult, NUM_ELEMENTS, 100);
    auto timeCross3 = std::chrono::high_resolution_clock::now();

    hce(cudaGetLastError());

    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    for(int i=0; i<NUM_ELEMENTS; i++) {
        if(length(cuResult[i] - make_float4(glmResult[i])) > 0.0001f) {
            std::cerr << "error cross i=" << i << std::endl;
            break;
        }
        if(glm::length(cpuResult[i] - glmResult[i]) > 0.01f) {
            std::cerr << "error cross i=" << i << std::endl;
            break;
        }
    }

    std::cout << "time for cpu glm (matrix):          " << std::chrono::duration_cast<std::chrono::milliseconds>(timeMatrix3 -  timeMatrix2).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda glm (matrix):         " << std::chrono::duration_cast<std::chrono::milliseconds>(timeMatrix1 -  timeMatrix0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (matrix): " << std::chrono::duration_cast<std::chrono::milliseconds>(timeMatrix2 -  timeMatrix1).count() << " milliseconds" << std::endl;

    std::cout << "time for cpu glm (dot):             " << std::chrono::duration_cast<std::chrono::milliseconds>(timeDot3 -     timeDot2).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda glm (dot):            " << std::chrono::duration_cast<std::chrono::milliseconds>(timeDot1 -     timeDot0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (dot):    " << std::chrono::duration_cast<std::chrono::milliseconds>(timeDot2 -     timeDot1).count() << " milliseconds" << std::endl;

    std::cout << "time for cpu glm (cross):           " << std::chrono::duration_cast<std::chrono::milliseconds>(timeCross3 -   timeCross2).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda glm (cross):          " << std::chrono::duration_cast<std::chrono::milliseconds>(timeCross1 -   timeCross0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (cross):  " << std::chrono::duration_cast<std::chrono::milliseconds>(timeCross2 -   timeCross1).count() << " milliseconds" << std::endl;

    delete[] glmVectors;
    delete[] cuVectors;
    delete[] cpuResult;
    delete[] glmResult;
    delete[] cuResult;

    cudaFree(d_glmVectors);
    cudaFree(d_glmResult);
    cudaFree(d_cuVectors);
    cudaFree(d_cuResult);

    return 0;
}

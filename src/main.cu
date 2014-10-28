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

__device__ __host__ glm::vec4 mul(glm::mat4 m, glm::vec4 v) {
    return glm::vec4(m[0].x*v.x + m[1].x*v.y + m[2].x*v.z + m[3].x*v.w,
                     m[0].y*v.x + m[1].y*v.y + m[2].y*v.z + m[3].y*v.w,
                     m[0].z*v.x + m[1].z*v.y + m[2].z*v.z + m[3].z*v.w,
                     m[0].w*v.x + m[1].w*v.y + m[2].w*v.z + m[3].w*v.w);
}

__global__ void cuKernel(const float4 *vectors, mat4 matrix, float4 *result, int numElements, int innerLoopSize, bool matrixTest) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        if(matrixTest) {
            result[i] = matrix * vectors[i];
            if(i > 3) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i] = matrix * vectors[i];
                    result[i] += matrix * vectors[i-1];
                    result[i] += matrix * vectors[i-2];
                    result[i] += matrix * vectors[i-3];
                    result[i] += matrix * vectors[i-4];
                }
            }
        }
        else {
            result[i] = vectors[i];
            if(i>1) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i].w = dot(vectors[i-1], vectors[i]);
                    float3 tmp = make_float3(result[i]);
                    float3 tmp2 = make_float3(vectors[i]);
                    tmp = cross(tmp, tmp2);
                    result[i].x = dot(tmp, tmp2);

                    tmp = make_float3(result[i]);
                    tmp2 = make_float3(vectors[i]);
                    tmp = cross(tmp, tmp2);
                    result[i].y = dot(tmp, tmp2);

                    tmp = make_float3(result[i]);
                    tmp2 = make_float3(vectors[i]);
                    tmp = cross(tmp, tmp2);
                    result[i].z = dot(tmp, tmp2);
                }
            }
        }
    }
}

__global__ void glmMulKernel(const glm::vec4 *vectors, const glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize, bool matrixTest) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        if(matrixTest) {
            result[i] = mul(matrix, vectors[i]);
            if(i > 3) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i] = mul(matrix, vectors[i]);
                    result[i] += mul(matrix, vectors[i-1]);
                    result[i] += mul(matrix, vectors[i-2]);
                    result[i] += mul(matrix, vectors[i-3]);
                    result[i] += mul(matrix, vectors[i-4]);
                }
            }
        }
        else {
            result[i] = vectors[i];
            if(i>1) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i].w = glm::dot(vectors[i-1], vectors[i]);
                    glm::vec3 tmp(result[i]);
                    glm::vec3 tmp2(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].x = glm::dot(tmp, tmp2);

                    tmp = glm::vec3(result[i]);
                    tmp2 = glm::vec3(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].y = glm::dot(tmp, tmp2);

                    tmp = glm::vec3(result[i]);
                    tmp2 = glm::vec3(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].z = glm::dot(tmp, tmp2);
                }
            }
        }
    }
}

__global__ void glmKernel(const glm::vec4 *vectors, const glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize, bool matrixTest) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        if(matrixTest) {
            result[i] = matrix * vectors[i];
            if(i > 3) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i] = matrix * vectors[i];
                    result[i] += matrix * vectors[i-1];
                    result[i] += matrix * vectors[i-2];
                    result[i] += matrix * vectors[i-3];
                    result[i] += matrix * vectors[i-4];
                }
            }
        }
        else {
            result[i] = vectors[i];
            if(i>1) {
                for(int j=0; j<innerLoopSize; j++) {
                    result[i].w = glm::dot(vectors[i-1], vectors[i]);
                    glm::vec3 tmp(result[i]);
                    glm::vec3 tmp2(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].x = glm::dot(tmp, tmp2);

                    tmp = glm::vec3(result[i]);
                    tmp2 = glm::vec3(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].y = glm::dot(tmp, tmp2);

                    tmp = glm::vec3(result[i]);
                    tmp2 = glm::vec3(vectors[i]);
                    tmp = glm::cross(tmp, tmp2);
                    result[i].z = glm::dot(tmp, tmp2);
                }
            }
        }
    }
}

void cpuKernel(const glm::vec4 *vectors, const glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize, bool matrixTest) {
    for(int i=0; i<numElements; i++) {
        if(i < numElements) {
            if(matrixTest) {
                result[i] = matrix * vectors[i];
                if(i > 3) {
                    for(int j=0; j<innerLoopSize; j++) {
                        result[i] = matrix * vectors[i];
                        result[i] += matrix * vectors[i-1];
                        result[i] += matrix * vectors[i-2];
                        result[i] += matrix * vectors[i-3];
                        result[i] += matrix * vectors[i-4];
                    }
                }
            }
            else {
                result[i] = vectors[i];
                if(i>1) {
                    for(int j=0; j<innerLoopSize; j++) {
                        result[i].w = glm::dot(vectors[i-1], vectors[i]);
                        glm::vec3 tmp(result[i]);
                        glm::vec3 tmp2(vectors[i]);
                        tmp = glm::cross(tmp, tmp2);
                        result[i].x = glm::dot(tmp, tmp2);

                        tmp = glm::vec3(result[i]);
                        tmp2 = glm::vec3(vectors[i]);
                        tmp = glm::cross(tmp, tmp2);
                        result[i].y = glm::dot(tmp, tmp2);

                        tmp = glm::vec3(result[i]);
                        tmp2 = glm::vec3(vectors[i]);
                        tmp = glm::cross(tmp, tmp2);
                        result[i].z = glm::dot(tmp, tmp2);
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    glm::mat4 glmMatrix;
    glmMatrix[0] = glm::vec4(3.14f, 15.f, .92f, .65f);
    glmMatrix[1] = glm::vec4(.35f, .89f, .79f, .32f);
    glmMatrix[2] = glm::vec4(.38f, .46f, .26f, .43f);
    glmMatrix[3] = glm::vec4(.38f, .32f, .79f, .50f);

    mat4 cuMatrix = make_mat4(glmMatrix);

    glm::vec4* glmVectors = new glm::vec4[NUM_ELEMENTS];
    float4* cuVectors = new float4[NUM_ELEMENTS];

    for(int i=0; i<NUM_ELEMENTS; i++) {
        glmVectors[i] = glm::vec4(rand() / (float) RAND_MAX, rand() / (float) RAND_MAX, rand() / (float) RAND_MAX, rand() / (float) RAND_MAX);
        cuVectors[i] = make_float4(glmVectors[i]);
    }

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
    glmKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 10, true);
    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 10, true);
    hce(cudaDeviceSynchronize());

    auto interim0 = std::chrono::high_resolution_clock::now();

    glmMulKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 100, true);
    hce(cudaDeviceSynchronize());
    auto interim1 = std::chrono::high_resolution_clock::now();

    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 100, true);
    hce(cudaDeviceSynchronize());
    auto interim2 = std::chrono::high_resolution_clock::now();

//    cpuKernel(glmVectors, glmMatrix, cpuResult, NUM_ELEMENTS, 100, true); // takes too long
//    auto interim3 = std::chrono::high_resolution_clock::now();

    glm::vec4* glmResult = new glm::vec4[NUM_ELEMENTS];
    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    float4* cuResult = new float4[NUM_ELEMENTS];
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    hce(cudaGetLastError());
    for(int i=0; i<NUM_ELEMENTS; i++) {
//        assert(glm::length(glmResult[i] - cuResult[i]) < 0.0001f);
//        assert(length(cuResult[i] - make_float4(cpuResult[i])) < 0.0001f);
        if(length(cuResult[i] - make_float4(glmResult[i])) > 0.000001f) {
            std::cerr << "error matrix i=" << i << std::endl;
            break;
        }
    }

    auto interim4 = std::chrono::high_resolution_clock::now();

    glmMulKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 100, false);
    hce(cudaDeviceSynchronize());
    auto interim5 = std::chrono::high_resolution_clock::now();

    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 100, false);
    hce(cudaDeviceSynchronize());
    auto interim6 = std::chrono::high_resolution_clock::now();

    hce(cudaGetLastError());

    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    for(int i=0; i<NUM_ELEMENTS; i++) {
//        assert(glm::length(glmResult[i] - cuResult[i]) < 0.0001f);
//        assert(length(cuResult[i] - make_float4(cpuResult[i])) < 0.0001f);
        if(length(cuResult[i] - make_float4(glmResult[i])) > 0.000001f) {
            std::cerr << "error cross dot i=" << i << std::endl;
            break;
        }
    }

    std::cout << "time for cuda glm (matrix): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim1 - interim0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (matrix): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim2 - interim1).count() << " milliseconds" << std::endl;
//    std::cout << "time for cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(interim3 - interim2).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda glm (dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim5 - interim4).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim6 - interim5).count() << " milliseconds" << std::endl;

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

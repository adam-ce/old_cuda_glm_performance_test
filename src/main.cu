/*
#include <QCoreApplication>
#include "opengltest.h"


int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    opengltest foo;
    return app.exec();
}
*/

#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "helper_math.h"
#include <stdio.h>
#include <chrono>

#define NUM_ELEMENTS 2000000
#define THREADS_PER_BLOCK 256

//handle cuda errors
void hce(cudaError_t error)
{
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

__global__ void cuKernel(const float4 *vectors, mat4 matrix, float4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = matrix * vectors[i];
        if(i > 3) {
            result[i] += matrix * vectors[i-1];
            result[i] += matrix * vectors[i-2];
            result[i] += matrix * vectors[i-3];
            result[i] += matrix * vectors[i-4];

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

__global__ void glmKernel(const glm::vec4 *vectors, const glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements) {
        result[i] = matrix * vectors[i];
        if(i > 3) {
            result[i] += matrix * vectors[i-1];
            result[i] += matrix * vectors[i-2];
            result[i] += matrix * vectors[i-3];
            result[i] += matrix * vectors[i-4];

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

void cpuKernel(const glm::vec4 *vectors, const glm::mat4 matrix, glm::vec4 *result, int numElements, int innerLoopSize) {
    for(int i=0; i<numElements; i++) {
        if(i < numElements) {
            result[i] = matrix * vectors[i];
            if(i > 3) {
                result[i] += matrix * vectors[i-1];
                result[i] += matrix * vectors[i-2];
                result[i] += matrix * vectors[i-3];
                result[i] += matrix * vectors[i-4];

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
    glmKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 10);
    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 10);
    hce(cudaDeviceSynchronize());

    auto interim0 = std::chrono::high_resolution_clock::now();

    glmKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 0);
    hce(cudaDeviceSynchronize());
    auto interim1 = std::chrono::high_resolution_clock::now();

    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 0);
    hce(cudaDeviceSynchronize());
    auto interim2 = std::chrono::high_resolution_clock::now();

//    cpuKernel(glmVectors, glmMatrix, cpuResult, NUM_ELEMENTS); // takes too long
//    auto interim3 = std::chrono::high_resolution_clock::now();

    auto interim4 = std::chrono::high_resolution_clock::now();

    glmKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_glmVectors, glmMatrix, d_glmResult, NUM_ELEMENTS, 100);
    hce(cudaDeviceSynchronize());
    auto interim5 = std::chrono::high_resolution_clock::now();

    cuKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_cuVectors, cuMatrix, d_cuResult, NUM_ELEMENTS, 100);
    hce(cudaDeviceSynchronize());
    auto interim6 = std::chrono::high_resolution_clock::now();

    hce(cudaGetLastError());

    glm::vec4* glmResult = new glm::vec4[NUM_ELEMENTS];
    hce(cudaMemcpy(glmResult, d_glmResult, glmSize, cudaMemcpyDeviceToHost));
    float4* cuResult = new float4[NUM_ELEMENTS];
    hce(cudaMemcpy(cuResult, d_cuResult, glmSize, cudaMemcpyDeviceToHost));

    for(int i=0; i<NUM_ELEMENTS; i++) {
//        assert(glm::length(glmResult[i] - cuResult[i]) < 0.0001f);
//        assert(length(cuResult[i] - make_float4(cpuResult[i])) < 0.0001f);
        assert(length(cuResult[i] - make_float4(glmResult[i])) < 0.0001f);
    }

    std::cout << "time for cuda glm (without dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim1 - interim0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (without dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim2 - interim1).count() << " milliseconds" << std::endl;
//    std::cout << "time for cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(interim3 - interim2).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda glm (with dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim5 - interim4).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda helper math (with dot and cross): " << std::chrono::duration_cast<std::chrono::milliseconds>(interim6 - interim5).count() << " milliseconds" << std::endl;

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

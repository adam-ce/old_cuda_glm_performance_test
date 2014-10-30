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

#define GLM_FORCE_RADIANS
#include <random>
#include <glm/glm.hpp>
#include <iostream>
#include <cuda_runtime.h>
//#include "helper_math.h"
//#include <stdio.h>
//#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

#define NUM_ELEMENTS 2000000

//handle cuda errors
void hce(cudaError_t error)
{
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

float randf(float upperBorder = 1.f) {
    return upperBorder * ((float) rand()) / (float) RAND_MAX;
}

struct CudaUniforms {
    __host__ CudaUniforms(int width, int height) {
        this->width = width;
        this->height = height;
        cameraPosition = glm::vec4(randf(10.f), randf(30.f), randf(30.f), 1.f);
        cameraInvVPMatrix = glm::perspective(glm::radians(randf(100.f)), glm::radians(randf(100.f)), 0.2f, 100.f) * glm::lookAt(glm::vec3(cameraPosition), glm::vec3(20.f, 15.f, 10.f), glm::vec3(0.f, 1.f, 0.f));
    }

    unsigned short width;
    unsigned short height;
    glm::mat4 cameraInvVPMatrix;
    glm::vec4 cameraPosition;
};
struct CudaLight {
    __host__ CudaLight() {
        colour = glm::vec3(randf(1.f), randf(1.0), randf(1.0));
        pos = glm::vec3(randf(10.f), randf(30.f), randf(30.f));
        vpMat = glm::perspective(glm::radians(randf(100.f)), glm::radians(randf(100.f)), 0.2f, 100.f) * glm::lookAt(pos, glm::vec3(20.f, 15.f, 10.f), glm::vec3(0.f, 1.f, 0.f));
    }
    glm::vec3 colour;
    glm::vec3 pos;
    glm::mat4 vpMat;
};

//__device__ CudaUniforms* g_uniforms;
//__device__ CudaLight* g_lights;
//__device__ int g_lightsCount;

//union UnpackToShortUnion{
//    unsigned int a;
//    struct {
//        short a, b;
//    } myShorts;
//};
//union UnpackToUCharUnion{
//    unsigned int a;
//    struct {
//        unsigned char a, b, c, d;
//    } myChars;
//};
//union UnpackToCharUnion{
//    unsigned int val;
//    struct {
//        char a, b, c, d;
//    } chars;
//};

//__device__ void unpackGeomData(const uint4& d0, glm::vec4* pos, int* materialIndex, glm::vec3* normal, glm::vec4* textureColour)
//{
//    float x = (float) (blockIdx.x*blockDim.x + threadIdx.x);
//    x /= (float) g_uniforms->width;
//    x = x * 2.f - 1.f;
//    float y = (float) (blockIdx.y*blockDim.y + threadIdx.y);
//    y /= (float) g_uniforms->height;
//    y = y * 2.f - 1.f;

//    glm::vec4 p1(x, y, -1.f, 1.f);
//    p1 = g_uniforms->cameraInvVPMatrix * p1;
//    p1 /= p1.w;
//    glm::vec4 p2(x, y, 1.f, 1.f);
//    p2 = g_uniforms->cameraInvVPMatrix * p2;
//    p2 /= p2.w;
//    glm::vec4 dir = p2 - p1;
//    dir.w = 0;
//    dir = glm::normalize(dir);

//    float dist = __int_as_float(d0.x);
//    *pos = g_uniforms->cameraPosition + dir * dist;
//    pos->w = 1.f;

//    *materialIndex = d0.w;

//    UnpackToCharUnion charUnpacker;
//    charUnpacker.val = d0.y;
//    normal->x = ((float) charUnpacker.chars.a) / 127.f;
//    normal->y = ((float) charUnpacker.chars.b) / 127.f;
//    normal->z = ((float) charUnpacker.chars.c) / 127.f;

//    UnpackToUCharUnion blib;
//    blib.a = d0.z;
//    textureColour->x = ((float) blib.myChars.a) / 255.f;
//    textureColour->y = ((float) blib.myChars.b) / 255.f;
//    textureColour->z = ((float) blib.myChars.c) / 255.f;
//    textureColour->w = 0.f;
//}

//__device__ uchar4 rgbaFloatToUChar4(glm::vec4 rgba)
//{
//    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
//    rgba.y = __saturatef(rgba.y);
//    rgba.z = __saturatef(rgba.z);
//    rgba.w = __saturatef(rgba.w);
//    return make_uchar4(rgba.x * 255.0f, rgba.y * 255.0f, rgba.z * 255.0f, rgba.w * 255.0f);
//}

//__global__ void glmKernel(const uint4* input, uchar4 *result) {
//    int x = (int) blockIdx.x*blockDim.x + threadIdx.x;
//    int y = (int) blockIdx.y*blockDim.y + threadIdx.y;
//    if(x >= g_uniforms->width) return;
//    if(y >= g_uniforms->height) return;

//    uint4 geomData0 = input[y * g_uniforms->width + x];

//    glm::vec4 pos, textureColour;
//    glm::vec3 normal;
//    int materialIndex;

//    unpackGeomData(geomData0, &pos, &materialIndex, &normal, &textureColour);
//    glm::vec4 outColour(0.f, 1.f, 1.f, 0.0f);

//    if (materialIndex < 50 && materialIndex > 0) {
//        outColour = glm::vec4(textureColour.x * 0.2f, textureColour.y * 0.2f, textureColour.z * 0.2f, 0.f);
//        for(int i=0; i<g_lightsCount; i++) {
//            glm::vec4 shadingPosInLightSpace = g_lights[i].vpMat * pos;
//            if(shadingPosInLightSpace.z < 0.f) continue;
//            shadingPosInLightSpace /= shadingPosInLightSpace.w;
//            if(shadingPosInLightSpace.x < -1.f || shadingPosInLightSpace.x > 1.f) continue;
//            if(shadingPosInLightSpace.y < -1.f || shadingPosInLightSpace.y > 1.f) continue;

//            glm::vec3 lightVec = g_lights[i].pos - glm::vec3(pos);
//            float lightDist = glm::length(lightVec);

//            lightVec /= lightDist;
//            float f = 100.f * glm::dot(normal, lightVec) / (lightDist*lightDist);
//            outColour += textureColour*f;
//        }
//    }

//    uchar4 data = rgbaFloatToUChar4(outColour);
//    result[y * g_uniforms->width + x] = data;
//}

#define LIGHT_COUNT 2
#define CUDA_BLOCK_WIDTH 4
#define CUDA_BLOCK_HEIGHT 4
#define WIDTH 2000
#define HEIGHT 2000

int main(int argc, char *argv[]) {
    const int width = WIDTH;
    const int height = HEIGHT;
//    CudaLight glmLights[LIGHT_COUNT];
//    CudaUniforms uniforms(width, height);

    uint4 input[WIDTH * HEIGHT];
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            uint4 d;
//            float dist = randf(10.f);
//            d.x = glm::floatBitsToUint(dist);
            glm::vec3 norm(randf(1.f), randf(1.f), randf(1.f));
            norm = glm::normalize(norm);
//            d.y = glm::packSnorm4x8(glm::vec4(norm.x, norm.y, norm.z, 1.f));
//            d.z = glm::packUnorm4x8(glm::vec4(randf(), randf(), randf(), randf()));
            d.w = 20;
            input[j*width + i] = d;
        }
    }

//    CudaLight* m_deviceCudaLights = 0;
//    hce(cudaMalloc((void**)&m_deviceCudaLights, sizeof(CudaLight) * LIGHT_COUNT ));
//    hce(cudaMemcpy(m_deviceCudaLights, &glmLights, sizeof(CudaLight) * LIGHT_COUNT, cudaMemcpyHostToDevice));
//    hce(cudaMemcpyToSymbol(g_lights, &m_deviceCudaLights, sizeof(CudaLight*)));
//    int lightCount = LIGHT_COUNT;
//    hce(cudaMemcpyToSymbol(g_lightsCount, &lightCount, sizeof(int)));

//    CudaUniforms* m_deviceUniforms = 0;
//    hce(cudaMalloc((void**)&m_deviceUniforms, sizeof(CudaUniforms) ));
//    hce(cudaMemcpy(m_deviceUniforms, &uniforms, sizeof(CudaUniforms), cudaMemcpyHostToDevice));
//    hce(cudaMemcpyToSymbol(g_uniforms, &m_deviceUniforms, sizeof(CudaUniforms*)));


//    dim3 threadsPerBlock(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);
//    dim3 numBlocks(width / CUDA_BLOCK_WIDTH, height / CUDA_BLOCK_HEIGHT);
////    dim3 numBlocks(100, 100);

//    if(width % CUDA_BLOCK_WIDTH > 0) numBlocks.x++;
//    if(height % CUDA_BLOCK_HEIGHT > 0) numBlocks.y++;


//    uint4* d_input;
//    hce(cudaMalloc((void**)&d_input, sizeof(uint4) * width * height ));
//    hce(cudaMemcpy(d_input, input, sizeof(uint4) * width * height, cudaMemcpyHostToDevice));

//    uchar4* d_output;
//    hce(cudaMalloc((void**)&d_output, sizeof(uchar4) * width * height ));
//    glmKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output);
//    hce(cudaDeviceSynchronize());

//    hce(cudaFree(d_input));
//    hce(cudaFree(d_output));
//    hce(cudaFree(m_deviceUniforms));
//    hce(cudaFree(m_deviceCudaLights));
    return 0;
}

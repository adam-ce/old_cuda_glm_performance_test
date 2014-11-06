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
#include <cstdlib>
#include <ctime>
#include <glm/glm.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <stdio.h>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

//handle cuda errors
void hce(cudaError_t error)
{
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

float randf(float upperBorder = 1.f) {
    int randi = std::rand();
    float rand1f = ((float) randi) / (float) RAND_MAX;
    return upperBorder * rand1f;
}

struct GlmUniforms {
    __host__ GlmUniforms(int width, int height) {
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
struct GlmLight {
    __host__ GlmLight() {
        colour = glm::vec3(randf(1.f), randf(1.0), randf(1.0));
        pos = glm::vec3(randf(10.f), randf(30.f), randf(30.f));
        vpMat = glm::perspective(glm::radians(randf(100.f)), glm::radians(randf(100.f)), 0.2f, 100.f) * glm::lookAt(pos, glm::vec3(20.f, 15.f, 10.f), glm::vec3(0.f, 1.f, 0.f));
    }
    glm::vec3 colour;
    glm::vec3 pos;
    glm::mat4 vpMat;
};
struct CudaUniforms {
    __host__ CudaUniforms(const GlmUniforms& ref) {
        this->width = ref.width;
        this->height = ref.height;
        cameraPosition = make_float4(ref.cameraPosition);
        cameraInvVPMatrix = make_mat4(ref.cameraInvVPMatrix);
    }

    unsigned short width;
    unsigned short height;
    mat4 cameraInvVPMatrix;
    float4 cameraPosition;
};
struct CudaLight {
    __host__ CudaLight(const GlmLight& ref) {
        colour = make_float3(ref.colour);
        pos = make_float3(ref.pos);
        vpMat = make_mat4(ref.vpMat);
    }
    float3 colour;
    float3 pos;
    mat4 vpMat;
};

__device__ GlmUniforms* g_glmUniforms;
__device__ GlmLight* g_glmLights;
__device__ CudaUniforms* g_cudaUniforms;
__device__ CudaLight* g_cudaLights;
__device__ int g_lightsCount;

union UnpackToShortUnion{
    unsigned int a;
    struct {
        short a, b;
    } myShorts;
};
union UnpackToUCharUnion{
    unsigned int a;
    struct {
        unsigned char a, b, c, d;
    } myChars;
};
union UnpackToCharUnion{
    unsigned int val;
    struct {
        char a, b, c, d;
    } chars;
};

#define GLM_NAMESPACE glm
namespace test {
//__device__ __forceinline__ glm::vec4 mul(glm::mat4 m, glm::vec4 v) {
//    return glm::mat4x4::col_type(
//        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
//        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
//        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
//        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3]);
//}

//__device__ __forceinline__ float dot(glm::vec4 a, glm::vec4 b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
//}
//__device__ __forceinline__ float dot(glm::vec3 a, glm::vec3 b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z;
//}

//__device__ __forceinline__ glm::vec4 normalize(glm::vec4 v) {
//    float invLen = rsqrtf(test::dot(v, v));
//    return v * invLen;
//}
//__device__ __forceinline__ glm::vec3 normalize(glm::vec3 v) {
//    float invLen = rsqrtf(test::dot(v, v));
//    return v * invLen;
//}

//__device__ __inline__ float length(glm::vec4 vec) {
//    return sqrtf(test::dot(vec, vec));
//}
//__device__ __inline__ float length(glm::vec3 vec) {
//    return sqrtf(test::dot(vec, vec));
//}

}

__device__ void unpackGeomDataGlm(const uint4& d0, glm::vec4* pos, int* materialIndex, glm::vec3* normal, glm::vec4* textureColour)
{
    float x = (float) (blockIdx.x*blockDim.x + threadIdx.x);
    x /= (float) g_glmUniforms->width;
    x = x * 2.f - 1.f;
    float y = (float) (blockIdx.y*blockDim.y + threadIdx.y);
    y /= (float) g_glmUniforms->height;
    y = y * 2.f - 1.f;

    glm::vec4 p1(x, y, -1.f, 1.f);
    p1 = g_glmUniforms->cameraInvVPMatrix * p1;
    p1 /= p1.w;
    glm::vec4 p2(x, y, 1.f, 1.f);
    p2 = g_glmUniforms->cameraInvVPMatrix * p2;
    p2 /= p2.w;
    glm::vec4 dir = p2 - p1;
    dir.w = 0;
    dir = GLM_NAMESPACE::normalize(dir);

    float dist = __int_as_float(d0.x);
    *pos = g_glmUniforms->cameraPosition + dir * dist;
    pos->w = 1.f;

    *materialIndex = d0.w;

    UnpackToCharUnion charUnpacker;
    charUnpacker.val = d0.y;
    normal->x = ((float) charUnpacker.chars.a) / 127.f;
    normal->y = ((float) charUnpacker.chars.b) / 127.f;
    normal->z = ((float) charUnpacker.chars.c) / 127.f;

    UnpackToUCharUnion blib;
    blib.a = d0.z;
    textureColour->x = ((float) blib.myChars.a) / 255.f;
    textureColour->y = ((float) blib.myChars.b) / 255.f;
    textureColour->z = ((float) blib.myChars.c) / 255.f;
    textureColour->w = 0.f;
}
__device__ void unpackGeomDataCuda(const uint4& d0, float4* pos, int* materialIndex, float3* normal, float4* textureColour)
{
    float x = (float) (blockIdx.x*blockDim.x + threadIdx.x);
    x /= (float) g_cudaUniforms->width;
    x = x * 2.f - 1.f;
    float y = (float) (blockIdx.y*blockDim.y + threadIdx.y);
    y /= (float) g_cudaUniforms->height;
    y = y * 2.f - 1.f;

    float4 p1 = make_float4(x, y, -1.f, 1.f);
    p1 = g_cudaUniforms->cameraInvVPMatrix * p1;
    p1 /= p1.w;
    float4 p2 = make_float4(x, y, 1.f, 1.f);
    p2 = g_cudaUniforms->cameraInvVPMatrix * p2;
    p2 /= p2.w;
    float4 dir = p2 - p1;
    dir.w = 0;
    dir = normalize(dir);

    float dist = __int_as_float(d0.x);
    *pos = g_cudaUniforms->cameraPosition + dir * dist;
    pos->w = 1.f;

    *materialIndex = d0.w;

    UnpackToCharUnion charUnpacker;
    charUnpacker.val = d0.y;
    normal->x = ((float) charUnpacker.chars.a) / 127.f;
    normal->y = ((float) charUnpacker.chars.b) / 127.f;
    normal->z = ((float) charUnpacker.chars.c) / 127.f;

    UnpackToUCharUnion blib;
    blib.a = d0.z;
    textureColour->x = ((float) blib.myChars.a) / 255.f;
    textureColour->y = ((float) blib.myChars.b) / 255.f;
    textureColour->z = ((float) blib.myChars.c) / 255.f;
    textureColour->w = 0.f;
}

__device__ uchar4 rgbaFloatToUChar4Glm(glm::vec4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return make_uchar4(rgba.x * 255.0f, rgba.y * 255.0f, rgba.z * 255.0f, rgba.w * 255.0f);
}

__device__ uchar4 rgbaFloatToUChar4Cuda(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return make_uchar4(rgba.x * 255.0f, rgba.y * 255.0f, rgba.z * 255.0f, rgba.w * 255.0f);
}

__global__ void glmKernel(const uint4* input, uchar4 *result) {
    int x = (int) blockIdx.x*blockDim.x + threadIdx.x;
    int y = (int) blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= g_glmUniforms->width) return;
    if(y >= g_glmUniforms->height) return;

    uint4 geomData0 = input[y * g_glmUniforms->width + x];

    glm::vec4 pos, textureColour;
    glm::vec3 normal;
    int materialIndex;

    unpackGeomDataGlm(geomData0, &pos, &materialIndex, &normal, &textureColour);
    glm::vec4 outColour(0.f, 1.f, 1.f, 0.0f);

    if (materialIndex < 50 && materialIndex > 0) {
        outColour = glm::vec4(textureColour.x * 0.2f, textureColour.y * 0.2f, textureColour.z * 0.2f, 0.f);
        for(int i=0; i<g_lightsCount; i++) {
            glm::vec4 shadingPosInLightSpace = g_glmLights[i].vpMat * pos;
            if(shadingPosInLightSpace.z < 0.f) continue;
            shadingPosInLightSpace /= shadingPosInLightSpace.w;
            if(shadingPosInLightSpace.x < -1.f || shadingPosInLightSpace.x > 1.f) continue;
            if(shadingPosInLightSpace.y < -1.f || shadingPosInLightSpace.y > 1.f) continue;

            glm::vec3 lightVec = g_glmLights[i].pos - glm::vec3(pos);
            float lightDist = GLM_NAMESPACE::length(lightVec);

            lightVec /= lightDist;
            float f = 100.f * GLM_NAMESPACE::dot(normal, lightVec) / (lightDist*lightDist);
            outColour += textureColour*f;
        }
    }

    uchar4 data = rgbaFloatToUChar4Glm(outColour);
    result[y * g_glmUniforms->width + x] = data;
}

__global__ void cudaKernel(const uint4* input, uchar4 *result) {
    int x = (int) blockIdx.x*blockDim.x + threadIdx.x;
    int y = (int) blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= g_cudaUniforms->width) return;
    if(y >= g_cudaUniforms->height) return;

    uint4 geomData0 = input[y * g_cudaUniforms->width + x];

    float4 pos, textureColour;
    float3 normal;
    int materialIndex;

    unpackGeomDataCuda(geomData0, &pos, &materialIndex, &normal, &textureColour);
    float4 outColour = make_float4(0.f, 1.f, 1.f, 0.0f);

    if (materialIndex < 50 && materialIndex > 0) {
        outColour = make_float4(textureColour.x * 0.2f, textureColour.y * 0.2f, textureColour.z * 0.2f, 0.f);
        for(int i=0; i<g_lightsCount; i++) {
            float4 shadingPosInLightSpace = g_cudaLights[i].vpMat * pos;
            if(shadingPosInLightSpace.z < 0.f) continue;
            shadingPosInLightSpace /= shadingPosInLightSpace.w;
            if(shadingPosInLightSpace.x < -1.f || shadingPosInLightSpace.x > 1.f) continue;
            if(shadingPosInLightSpace.y < -1.f || shadingPosInLightSpace.y > 1.f) continue;

            float3 lightVec = g_cudaLights[i].pos - make_float3(pos);
            float lightDist = length(lightVec);

            lightVec /= lightDist;
            float f = 100.f * dot(normal, lightVec) / (lightDist*lightDist);
            outColour += textureColour*f;
        }
    }

    uchar4 data = rgbaFloatToUChar4Cuda(outColour);
    result[y * g_cudaUniforms->width + x] = data;
}

#define CUDA_BLOCK_WIDTH 8
#define CUDA_BLOCK_HEIGHT 4
#define WIDTH 5000
#define HEIGHT 5000
#define LIGHT_COUNT 2

int main(int argc, char *argv[]) {
    const int width = WIDTH;
    const int height = HEIGHT;
    std::srand(5845530);
    GlmLight glmLights[2];
    GlmUniforms glmUniforms(width, height);
    CudaLight cudaLights[] = {glmLights[0], glmLights[1]};
    CudaUniforms cudaUniforms(glmUniforms);

    uint4* input = new uint4[WIDTH * HEIGHT];

    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
//            std::cout << "i" << i << " j" << j << std::endl;
            uint4 data;
            float dist = randf(10.f);
            data.x = glm::floatBitsToUint(dist);
            float x, y, z;
            x=randf(1.f);
            y=randf(1.f);
            z=randf(1.f);
            glm::vec3 norm(x, y, z);
            norm = glm::normalize(norm);
            data.y = glm::packSnorm4x8(glm::vec4(norm.x, norm.y, norm.z, 1.f));
            int a, b, c, d;
            a = randf();
            b = randf();
            c = randf();
            d = randf();
            data.z = glm::packUnorm4x8(glm::vec4(a, b, c, d));
            data.w = 20;
            input[j*width + i] = data;
//            std::cout << "input: " << input[j*width + i].x << "/" << input[j*width + i].y << "/" << input[j*width + i].z << "/" << input[j*width + i].w << std::endl;
        }
    }

    GlmLight* m_deviceGlmLights = 0;
    hce(cudaMalloc((void**)&m_deviceGlmLights, sizeof(GlmLight) * LIGHT_COUNT ));
    hce(cudaMemcpy(m_deviceGlmLights, &glmLights, sizeof(GlmLight) * LIGHT_COUNT, cudaMemcpyHostToDevice));
    hce(cudaMemcpyToSymbol(g_glmLights, &m_deviceGlmLights, sizeof(GlmLight*)));
    int lightCount = LIGHT_COUNT;
    hce(cudaMemcpyToSymbol(g_lightsCount, &lightCount, sizeof(int)));

    GlmUniforms* m_deviceGlmUniforms = 0;
    hce(cudaMalloc((void**)&m_deviceGlmUniforms, sizeof(GlmUniforms) ));
    hce(cudaMemcpy(m_deviceGlmUniforms, &glmUniforms, sizeof(GlmUniforms), cudaMemcpyHostToDevice));
    hce(cudaMemcpyToSymbol(g_glmUniforms, &m_deviceGlmUniforms, sizeof(GlmUniforms*)));

    CudaLight* m_deviceCudaLights = 0;
    hce(cudaMalloc((void**)&m_deviceCudaLights, sizeof(CudaLight) * LIGHT_COUNT ));
    hce(cudaMemcpy(m_deviceCudaLights, &cudaLights, sizeof(CudaLight) * LIGHT_COUNT, cudaMemcpyHostToDevice));
    hce(cudaMemcpyToSymbol(g_cudaLights, &m_deviceCudaLights, sizeof(CudaLight*)));

    CudaUniforms* m_deviceCudaUniforms = 0;
    hce(cudaMalloc((void**)&m_deviceCudaUniforms, sizeof(CudaUniforms) ));
    hce(cudaMemcpy(m_deviceCudaUniforms, &cudaUniforms, sizeof(CudaUniforms), cudaMemcpyHostToDevice));
    hce(cudaMemcpyToSymbol(g_cudaUniforms, &m_deviceCudaUniforms, sizeof(CudaUniforms*)));


    dim3 threadsPerBlock(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);
    dim3 numBlocks(width / CUDA_BLOCK_WIDTH, height / CUDA_BLOCK_HEIGHT);
//    dim3 numBlocks(100, 100);

    if(width % CUDA_BLOCK_WIDTH > 0) numBlocks.x++;
    if(height % CUDA_BLOCK_HEIGHT > 0) numBlocks.y++;


    uint4* d_input;
    hce(cudaMalloc((void**)&d_input, sizeof(uint4) * width * height ));
    hce(cudaMemcpy(d_input, input, sizeof(uint4) * width * height, cudaMemcpyHostToDevice));

    uchar4* d_glmOutput;
    hce(cudaMalloc((void**)&d_glmOutput, sizeof(uchar4) * width * height ));
    glmKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_glmOutput);
    hce(cudaDeviceSynchronize());

    auto time0 = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < 10; r++) {
        glmKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_glmOutput);
        hce(cudaDeviceSynchronize());
    }
    auto time1 = std::chrono::high_resolution_clock::now();

    uchar4* d_cudaOutput;
    hce(cudaMalloc((void**)&d_cudaOutput, sizeof(uchar4) * width * height));
    cudaKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_cudaOutput);
    hce(cudaDeviceSynchronize());

    auto time2 = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < 10; r++) {
        cudaKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_cudaOutput);
        hce(cudaDeviceSynchronize());
    }
    auto time3 = std::chrono::high_resolution_clock::now();

    uchar4* glmOutput = new uchar4[sizeof(uchar4) * width * height];
    hce(cudaMemcpy(glmOutput, d_glmOutput, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
    uchar4* cudaOutput = new uchar4[sizeof(uchar4) * width * height];
    hce(cudaMemcpy(cudaOutput, d_cudaOutput, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    for(int i=0; i<width; i++) {
        for(int j=0; j<height; j++) {
            if(glmOutput[j*width + i] != cudaOutput[j*width + i]) {
                std::cout << "output not equal" << std::endl;
                i=width;
                break;
            }
        }
    }


    std::cout << "time for glm:   " << std::chrono::duration_cast<std::chrono::milliseconds>(time1 -  time0).count() << " milliseconds" << std::endl;
    std::cout << "time for cuda:  " << std::chrono::duration_cast<std::chrono::milliseconds>(time3 -  time2).count() << " milliseconds" << std::endl;

    hce(cudaFree(d_input));
    hce(cudaFree(d_glmOutput));
    hce(cudaFree(m_deviceGlmUniforms));
    hce(cudaFree(m_deviceGlmLights));

    hce(cudaFree(d_cudaOutput));
    hce(cudaFree(m_deviceCudaUniforms));
    hce(cudaFree(m_deviceCudaLights));
    delete[] input;
    delete[] glmOutput;
    delete[] cudaOutput;
    return 0;
}

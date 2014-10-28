project(cudaGlmPerfTest)
cmake_minimum_required(VERSION 2.6)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  link_directories(/usr/lib /usr/local/lib)

  # otherwise some bullet internal headers don't find friends..
  include_directories(/usr/local/cuda/include)
else()
  #windows
  #include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../winlib/include)
  #link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../winlib/lib)
endif()

set(project_SRCS
main.cu
)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math -gencode arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--use_fast_math -gencode arch=compute_20,code=sm_20")


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(CMAKE_CXX_FLAGS "-g -Wall -D _DEBUG -D _LINUX -std=c++11")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " -std=c++11")
else()
    SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}" )
endif()

cuda_add_executable(cudaGlmPerfTest ${project_SRCS})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIBS ${LIBS})
	target_link_libraries(cudaGlmPerfTest ${LIBS})
else()
    set(LIBS ${LIBS})
endif()



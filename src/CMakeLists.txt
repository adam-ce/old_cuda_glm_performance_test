#
# Copyright (C) 2014 Adam Celarek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math -gencode arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g -use_fast_math -gencode arch=compute_20,code=sm_20")


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(CMAKE_CXX_FLAGS "-g -Wall -D _DEBUG -D _LINUX -std=c++11")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " -std=c++11")
else()
    SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}" )
endif()

cuda_add_executable(test1 test1.cu)
cuda_add_executable(test2a test2a.cu)
cuda_add_executable(test2b test2b.cu)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIBS ${LIBS})
    target_link_libraries(test1 ${LIBS})
    target_link_libraries(test2a ${LIBS})
    target_link_libraries(test2b ${LIBS})
else()
    set(LIBS ${LIBS})
endif()



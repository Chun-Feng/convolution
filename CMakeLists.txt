cmake_minimum_required(VERSION 2.8)
project(conv)

find_package(CUDA)

cuda_add_executable(conv conv.cu)

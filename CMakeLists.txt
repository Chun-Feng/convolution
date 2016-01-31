cmake_minimum_required(VERSION 2.8)
project(conv)

find_package(CUDA)

# input dimensions
set(CONV_IMAGE_COUNT 128 CACHE STRING "number of input images")
set(CONV_IMAGE_WIDTH 32 CACHE STRING "width of the input image")
set(CONV_IMAGE_HEIGHT 32 CACHE STRING "height of the input image")
set(CONV_IMAGE_FEATURES 3 CACHE STRING "number of input features")
set(CONV_FILTER_SIZE 5 CACHE STRING
	"size of the filter (width and height are the same)")
set(CONV_FILTER_COUNT 64 CACHE STRING
	"number of filter size (same as the number of output features)")

# CUDA kernel options
set(CONV_IMAGES_PER_THREAD 4 CACHE STRING "")
set(CONV_FILTERS_PER_THREAD 4 CACHE STRING "")
set(CONV_THREADS_X 32 CACHE STRING "") # images
set(CONV_THREADS_Y 4 CACHE STRING "")  # filters
set(CONV_CACHED_PIXELS 4 CACHE STRING "")

configure_file("${PROJECT_SOURCE_DIR}/config.h.in"
	"${PROJECT_BINARY_DIR}/config.h")
include_directories(${PROJECT_BINARY_DIR})

cuda_add_executable(conv conv.cu)

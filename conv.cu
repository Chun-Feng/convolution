#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "config.h"

using namespace std;

#define CUDA_CHECK(ret) do { \
	cudaError_t errorcode = (ret); \
	if (errorcode != cudaSuccess) { \
		std::cout << "cuda error at file " << __FILE__ << " line " << __LINE__ \
			<< ": " << cudaGetErrorString(errorcode) << std::endl; \
		exit(1); \
	}} while(0)

// compute ceiling(x/y)
#define DIV_CEILING(x, y) (((x)+(y)-1)/(y))

// returns t2 - t1 in microseconds
int timespec_diff_us(timespec& t1, timespec& t2)
{
	return (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;
}

struct ConvolutionArguments
{
	float *images;
	int image_count;
	int image_width;
	int image_height;
	int image_features;

	float *filters;
	int filter_size;
	int filter_count;

	float *outputs;
};

// Don't try to optimize This function because it is only used to verify the
// GPU results.
int convolution_cpu(const ConvolutionArguments &args)
{
	const int image_size = args.image_width * args.image_height;
	const int filter_pixels = args.filter_size * args.filter_size;
	const int stride = args.image_count;

	timespec time_begin, time_end;
	clock_gettime(CLOCK_REALTIME, &time_begin);

	for (int i_img = 0; i_img < args.image_count; i_img++) {
		for (int i_filter = 0; i_filter < args.filter_count; i_filter++) {

			// convolution between one image feature and one filter
			for (int row = 0; row < args.image_height; row++) {
				for (int col = 0; col < args.image_width; col++) {
					float sum = 0.0;
					for (int frow = 0; frow < args.filter_size; frow++) {
						for (int fcol = 0; fcol < args.filter_size; fcol++) {
							int findex = frow * args.filter_size + fcol;
							int irow = row + frow - args.filter_size / 2;
							int icol = col + fcol - args.filter_size / 2;
							if (irow >= 0 && irow < args.image_height
									&& icol >= 0 && icol < args.image_width) {
								for (int i_feat = 0; i_feat < args.image_features; i_feat++) {
									sum +=
										args.images[i_feat * image_size * stride + (irow * args.image_width + icol) * stride + i_img] *
										args.filters[i_feat * filter_pixels * args.filter_count + findex * args.filter_count + i_filter];
								}
							}
						}
					}
					args.outputs[i_filter * image_size * stride + (row * args.image_width + col) * stride + i_img] = sum;
				}
			}
		}
	}

	clock_gettime(CLOCK_REALTIME, &time_end);
	return timespec_diff_us(time_begin, time_end);
}

template <int threads_x, int threads_y, int cached_pixels,
			int image_features, int images_per_thread,
			int filters_per_thread>
__global__
void convolution_kernel(const float *images, const float *filters,
		float *outputs, int image_count, int image_width, int image_height,
		int filter_size, int filter_count)
{
	const int image_pixels = image_width * image_height;
	const int filter_pixels = filter_size * filter_size;
	const int filters_per_block = threads_y * filters_per_thread;

	const int image_index = blockIdx.x * threads_x * images_per_thread + threadIdx.x;
	const int blocks_per_pixel = filter_count / (threads_y * filters_per_thread);
	const int filter_index = blockIdx.y % blocks_per_pixel;
	const int image_pixel_index = blockIdx.y / blocks_per_pixel;
	const int image_pixel_x = image_pixel_index % image_width;
	const int image_pixel_y = image_pixel_index / image_width;

	const int thread_id = threadIdx.y * THREADS_X + threadIdx.x;
	const int load_filter_pixel_index = thread_id / filters_per_block;
	const int load_filter_index = thread_id % filters_per_block;

	__shared__ float
		shm_images[cached_pixels * image_features][threads_x * images_per_thread];
	__shared__ float
		shm_filters[cached_pixels * image_features][threads_y * filters_per_thread];

	float result[filters_per_thread][images_per_thread];
	for (int f = 0; f < filters_per_thread; f++) {
		for (int g = 0; g < images_per_thread; g++) {
			result[f][g] = 0;
		}
	}

	// point to images(+0, +0, +0, +image_index)
	images += image_index;

	// point to filters(+0, +load_filter_pixel_index,
	//   +(filter_index * filters_per_block + load_filter_index))
	filters += load_filter_pixel_index * filter_count
		+ (filter_index * filters_per_block + load_filter_index);

	for (int pixel = 0; pixel < filter_pixels; pixel += cached_pixels) {
		// load data from global memory to shm_images
		for (int p = 0; p < cached_pixels; p += threads_y) {
			const int pixel_cache_index = p + threadIdx.y;
			const int pixel_index = pixel + pixel_cache_index;

			if (pixel_cache_index < cached_pixels) {
				const int x = image_pixel_x + pixel_index % filter_size - filter_size / 2;
				const int y = image_pixel_y + pixel_index / filter_size - filter_size / 2;
				const int image_base_index = (y * image_width + x) * image_count;
				if (y >= 0 && y < image_height && x >= 0 && x < image_width) {
					for (int f = 0; f < image_features; f++) {
						for (int i = 0; i < images_per_thread; i++) {
							if (image_index + i * cached_pixels < image_count)
								shm_images[f * cached_pixels + pixel_cache_index][threadIdx.x * images_per_thread + i]
									= images[image_base_index + f * image_pixels * image_count + i * threads_x];
							else
								shm_images[f * cached_pixels + pixel_cache_index][threadIdx.x * images_per_thread + i]
									= 0;
						}
					}
				} else {
					for (int i = 0; i < images_per_thread; i++) {
						for (int f = 0; f < image_features; f++) {
							shm_images[f * cached_pixels + pixel_cache_index][threadIdx.x * images_per_thread + i]
								= 0;
						}
					}
				}
			}
		}

		// load data from global memory to shm_filters
		if (load_filter_pixel_index < threads_x / filters_per_thread) {
			for (int p2 = 0; p2 < cached_pixels; p2 += threads_x / filters_per_thread) {
				if (p2 + load_filter_pixel_index < cached_pixels) {
					if (pixel + p2 + load_filter_pixel_index < filter_pixels) {
						for (int f = 0; f < image_features; f++) {
							shm_filters[f * cached_pixels + p2 + load_filter_pixel_index][load_filter_index]
								= filters[f * filter_pixels * filter_count + (pixel + p2) * filter_count];
						}
					} else {
						for (int f = 0; f < image_features; f++) {
							shm_filters[f * cached_pixels + p2 + load_filter_pixel_index][load_filter_index]
								= 0;
						}
					}
				}
			}
		}

		__syncthreads();

		// compute partial sums
		for (int i = 0; i < cached_pixels * image_features; i++) {
			for (int f = 0; f < filters_per_thread; f++) {
				for (int g = 0; g < images_per_thread; g++) {
					result[f][g] +=
						shm_images[i][threadIdx.x * images_per_thread + g]
						* shm_filters[i][threadIdx.y * filters_per_thread + f];
				}
			}
		}

		__syncthreads();
	}

	// NOTE: image_pixel_index = image_y * image_width + image_x
	outputs += (filter_index * filters_per_block + threadIdx.y * filters_per_thread)
			* image_pixels * image_count
		+ image_pixel_index * image_count
		+ image_index;

	for (int g = 0; g < images_per_thread; g++) {
		if (image_index + g * threads_x < image_count) {
			for (int f = 0; f < filters_per_thread; f++) {
				outputs[f * image_pixels * image_count + g * threads_x] = result[f][g];
			}
		}
	}
}

int convolution_gpu(const ConvolutionArguments &args)
{
	const int images_size = args.image_count * args.image_width *
		args.image_height * args.image_features;
	const int filters_size = args.filter_count * args.image_features *
		args.filter_size * args.filter_size;
	const int outputs_size = args.image_count * args.image_width *
		args.image_height * args.filter_count;
	const int image_pixels = args.image_width * args.image_height;

	// allocate device memory
	float *d_images, *d_filters, *d_outputs;
	CUDA_CHECK(cudaMalloc(&d_images, sizeof(float) * images_size));
	CUDA_CHECK(cudaMalloc(&d_filters, sizeof(float) * filters_size));
	CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(float) * outputs_size));

	// copy data to device
	CUDA_CHECK(cudaMemcpy(d_images, args.images, sizeof(float) * images_size,
				cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_filters, args.filters, sizeof(float) * filters_size,
				cudaMemcpyHostToDevice));

	dim3 dimGrid(DIV_CEILING(args.image_count, THREADS_X * IMAGES_PER_THREAD),
			(image_pixels * args.filter_count) / (THREADS_Y * FILTERS_PER_THREAD));
	dim3 dimBlock(THREADS_X, THREADS_Y);

	timespec time_begin, time_end;
	clock_gettime(CLOCK_REALTIME, &time_begin);

	convolution_kernel<THREADS_X, THREADS_Y, CACHED_PIXELS,
		TEST_IMAGE_FEATURES, IMAGES_PER_THREAD,
		FILTERS_PER_THREAD>
		<<<dimGrid, dimBlock>>>(
			d_images, d_filters, d_outputs,
			args.image_count, args.image_width, args.image_height,
			args.filter_size,
			args.filter_count);
	CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize(); // wait until convolution_kernel is finished

	clock_gettime(CLOCK_REALTIME, &time_end);

	// copy data from device
	CUDA_CHECK(cudaMemcpy(args.outputs, d_outputs, sizeof(float) * outputs_size,
				cudaMemcpyDeviceToHost));

	cudaFree(&d_images);
	cudaFree(&d_filters);
	cudaFree(&d_outputs);

	return timespec_diff_us(time_begin, time_end);
}

int main(int argc, char *argv[])
{
	ConvolutionArguments args;

	args.image_count = TEST_IMAGE_COUNT;
	args.image_width = TEST_IMAGE_WIDTH;
	args.image_height = TEST_IMAGE_HEIGHT;
	args.image_features = TEST_IMAGE_FEATURES;
	args.filter_size = TEST_FILTER_SIZE;
	args.filter_count = TEST_FILTERS;

	cout << "initializing" << endl;

	// allocate memory
	const int images_size = args.image_count * args.image_width *
		args.image_height * args.image_features;
	const int filters_size = args.filter_count * args.filter_size *
		args.filter_size * args.filter_size;
	const int outputs_size = args.image_count * args.image_width *
		args.image_height * args.filter_count;
	float *images = new float[images_size];
	float *filters = new float[filters_size];
	float *outputs_cpu = new float[outputs_size];
	float *outputs_gpu = new float[outputs_size];

	// initialize inputs
	srand(42);
	for (int i = 0; i < images_size; i++)
		images[i] = (rand() % 100 - 50) / 50.0;
	for (int i = 0; i < filters_size; i++)
		filters[i] = (rand() % 100 - 50) / 50.0;

	cout << "image: count=" << args.image_count
		<< " features=" << args.image_features
		<< " width=" << args.image_width
		<< " height=" << args.image_height << endl;
	cout << "filter: count=" << args.filter_count
		<< " size=" << args.filter_size << endl;
	cout << "output: count=" << args.image_count
		<< " features=" << args.filter_count
		<< " width=" << args.image_width
		<< " height=" << args.image_height << endl;

	cudaSetDevice(0); // this is used to initialize a GPU context

	args.images = images;
	args.filters = filters;

	cout << "running cpu convolution" << endl;
	args.outputs = outputs_cpu;
	int duration_cpu_us = convolution_cpu(args);

	cout << "running gpu convolution" << endl;
	args.outputs = outputs_gpu;
	int duration_gpu_us = convolution_gpu(args);

	// compare cpu and gpu answers
	float threshold = 0.00001;
	for (int i = 0; i < outputs_size; i++) {
		if (abs(outputs_cpu[i] - outputs_gpu[i]) >= threshold) {
			cout << "error: answers don't match at index " << i << endl;
			cout << "cpu output: " << outputs_cpu[i] << endl;
			for (int k = 0; k < 16; k++)
				cout << outputs_cpu[i + k] << " ";
			cout << endl;
			cout << "gpu output: " << outputs_gpu[i] << endl;
			for (int k = 0; k < 16; k++)
				cout << outputs_gpu[i + k] << " ";
			cout << endl << "full gpu output:" << endl;
			for (int k = 0; k < outputs_size; k++)
				cout << outputs_gpu[k] << " ";
			cout << endl;
			exit(1);
		}
	}
	cout << "compare ok" << endl;

	cout << "cpu duration: " << (duration_cpu_us / 1000.0) << " ms" << endl;
	cout << "gpu duration: " << (duration_gpu_us / 1000.0) << " ms" << endl;

	// "read" outputs (for valgrind checking)
	volatile int sink;
	for (int i = 0; i < outputs_size; i++)
		sink = outputs_cpu[i];

	// free memory
	delete[] images;
	delete[] filters;
	delete[] outputs_cpu;
	delete[] outputs_gpu;

	return 0;
}

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

using namespace std;

// FIXME: the code assumes filter size equals tiling size
#define TILE_X 5
#define TILE_Y 5

#define CUDA_CHECK(ret) do { \
	int errorcode = (ret); \
	if (errorcode != cudaSuccess) { \
		std::cout << "cuda error at file " << __FILE__ << " line " << __LINE__ \
			<< ": " << errorcode << std::endl; \
		exit(1); \
	}} while(0)

// returns t2 - t1 in milliseconds
int timespec_diff_ms(timespec& t1, timespec& t2)
{
	return (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_nsec - t1.tv_nsec) / 1e6;
}

struct ConvolutionArguments
{
	float *images;
	int image_count;
	int image_width;
	int image_height;
	int image_features;

	float *filters;
	int filter_width;
	int filter_height;

	float *outputs;
	int output_features;
};

int convolution_cpu(const ConvolutionArguments &args)
{
	const int image_size = args.image_width * args.image_height;
	const int filter_size = args.filter_width * args.filter_height;
	const int output_size = image_size;

	timespec time_begin, time_end;
	clock_gettime(CLOCK_REALTIME, &time_begin);

	for (int i_img = 0; i_img < args.image_count; i_img++) {
		for (int i_feat = 0; i_feat < args.image_features; i_feat++) {
			const int image_index = i_img * args.image_features + i_feat;
			const float *image = &args.images[image_index * image_size];
			for (int i_out_feat = 0; i_out_feat < args.output_features; i_out_feat++) {
				const int filter_index = i_feat * args.output_features + i_out_feat;
				const int output_index = i_img * args.output_features + i_out_feat;

				// convolution between one image feature and one filter
				const float *filter = &args.filters[filter_index * filter_size];
				float *output = &args.outputs[output_index * output_size];

				for (int row = 0; row < args.image_height; row++) {
					for (int col = 0; col < args.image_width; col++) {
						float sum = 0.0;
						for (int frow = 0; frow < args.filter_height; frow++) {
							for (int fcol = 0; fcol < args.filter_width; fcol++) {
								int irow = row + frow - args.filter_height / 2;
								int icol = col + fcol - args.filter_width / 2;
								if (irow >= 0 && irow < args.image_height
										&& icol >= 0 && icol < args.image_width) {
									sum += image[irow * args.image_width + icol] *
										filter[frow * args.filter_width + fcol];
								}
							}
						}
						output[row * args.image_width + col] = sum;
					}
				}
			}
		}
	}

	clock_gettime(CLOCK_REALTIME, &time_end);
	return timespec_diff_ms(time_begin, time_end);
}

__global__
void convolution_kernel(const float *images, const float *filters,
		float *outputs, int image_count, int image_width, int image_height,
		int image_features, int filter_width, int filter_height,
		int output_features)
{
	const int image_size = image_width * image_height;
	const int filter_size = filter_width * filter_height;
	const int output_size = image_size;

	const int col = blockIdx.x * TILE_X + threadIdx.x;
	const int row = blockIdx.y * TILE_Y + threadIdx.y;
	const int i_img = blockIdx.z;

	const float *image = &images[i_img * image_size];

	if (col < image_width && row < image_height) {
		for (int i_feat = 0; i_feat < image_features; i_feat++) {
			for (int i_out_feat = 0; i_out_feat < output_features; i_out_feat++) {
				const int filter_index = i_feat * output_features + i_out_feat;
				const int output_index = i_img * output_features + i_out_feat;

				const float *filter = &filters[filter_index * filter_size];
				float *output = &outputs[output_index * output_size];

				float sum = 0.0;
				for (int frow = 0; frow < filter_height; frow++) {
					for (int fcol = 0; fcol < filter_width; fcol++) {
						int irow = row + frow - filter_height / 2;
						int icol = col + fcol - filter_width / 2;
						if (irow >= 0 && irow < image_height
								&& icol >= 0 && icol < image_width) {
							sum += image[irow * image_width + icol] *
								filter[frow * filter_width + fcol];
						}
					}
				}
				output[row * image_width + col] = sum;
			}
		}
	}
}

int convolution_gpu(const ConvolutionArguments &args)
{
	const int images_size = args.image_count * args.image_width *
		args.image_height * args.image_features;
	const int filters_size = args.output_features * args.image_features *
		args.filter_width * args.filter_height;
	const int outputs_size = args.image_count * args.image_width *
		args.image_height * args.output_features;

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

	dim3 dimGrid(args.image_width / TILE_X + 1, args.image_height / TILE_Y + 1,
			args.image_count);
	dim3 dimBlock(TILE_X, TILE_Y, 1);

	timespec time_begin, time_end;
	clock_gettime(CLOCK_REALTIME, &time_begin);

	convolution_kernel<<<dimGrid, dimBlock>>>(d_images, d_filters, d_outputs,
			args.image_count, args.image_width, args.image_height,
			args.image_features, args.filter_width, args.filter_height,
			args.output_features);
	cudaDeviceSynchronize(); // wait until convolution_kernel is finished

	clock_gettime(CLOCK_REALTIME, &time_end);

	// copy data from device
	CUDA_CHECK(cudaMemcpy(args.outputs, d_outputs, sizeof(float) * outputs_size,
				cudaMemcpyDeviceToHost));

	cudaFree(&d_images);
	cudaFree(&d_filters);
	cudaFree(&d_outputs);

	return timespec_diff_ms(time_begin, time_end);
}

int main(int argc, char *argv[])
{
	bool gpu = false;

	if (argc == 2 && argv[1][0] == 'g')
		gpu = true;

	// set arguments

	ConvolutionArguments args;

	args.image_count = 32;
	args.image_width = 128;
	args.image_height = 128;
	args.image_features = 3;

	args.filter_width = TILE_X;
	args.filter_height = TILE_Y;

	args.output_features = 32;

	cout << "initializing" << endl;

	// allocate memory
	const int images_size = args.image_count * args.image_width *
		args.image_height * args.image_features;
	const int filters_size = args.output_features * args.image_features *
		args.filter_width * args.filter_height;
	const int outputs_size = args.image_count * args.image_width *
		args.image_height * args.output_features;
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
	cout << "filter: count=" << (args.image_features * args.output_features)
		<< " width=" << args.image_width
		<< " height=" << args.image_height << endl;
	cout << "output: count=" << args.image_count
		<< " features=" << args.output_features
		<< " width=" << args.image_width
		<< " height=" << args.image_height << endl;

	cudaSetDevice(0); // this is used to initialize a GPU context

	args.images = images;
	args.filters = filters;

	cout << "running cpu convolution" << endl;
	args.outputs = outputs_cpu;
	int duration_cpu = convolution_cpu(args);

	cout << "running gpu convolution" << endl;
	args.outputs = outputs_gpu;
	int duration_gpu = convolution_gpu(args);

	// compare cpu and gpu answers
	float threshold = 0.0001;
	//for (int i = 0; i < outputs_size; i++) {
	//	if (abs(outputs_cpu[i] - outputs_gpu[i]) >= threshold) {
	//		cout << "error: answers don't match at index " << i << endl;
	//		cout << outputs_cpu[i] << endl;
	//		cout << outputs_gpu[i] << endl;
	//		exit(1);
	//	}
	//}

	cout << "cpu duration: " << duration_cpu << " ms" << endl;
	cout << "gpu duration: " << duration_gpu << " ms" << endl;

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

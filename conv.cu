#include <cstdlib>

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

void convolution_cpu(const ConvolutionArguments &args)
{
	const int image_size = args.image_width * args.image_height;
	const int filter_size = args.filter_width * args.filter_height;
	const int output_size = image_size;

	for (int i_img = 0; i_img < args.image_count; i_img++) {
		for (int i_feat = 0; i_feat < args.image_features; i_feat++) {
			const int image_index = i_img * args.image_features + i_feat;
			const float *image = &args.images[image_index * image_size];
			for (int i_out_feat = 0; i_out_feat < args.output_features; i_out_feat++) {
				const int filter_index = i_feat * args.output_features + i_out_feat;
				const int output_index = i_img * args.output_features + i_out_feat;

				// convolution between one image feature and one filter
				const float *filter = &args.filters[filter_index * filter_size];
				float *output = &args.outputs[output_index * output_size]; // TODO: indexing

				for (int row = 0; row < args.image_height; row++) {
					for (int col = 0; col < args.image_width; col++) {
						float sum = 0.0;
						for (int frow = 0; frow < args.filter_width; frow++) {
							for (int fcol = 0; fcol < args.filter_height; fcol++) {
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
}

float generate_random_number()
{
	return (rand() % 100 - 50) / 50.0;
}

int main(int argc, char *argv[])
{
	// set arguments

	ConvolutionArguments args;

	args.image_count = 10;
	args.image_width = 64;
	args.image_height = 64;
	args.image_features = 3;

	args.filter_width = 5;
	args.filter_height = 5;

	args.output_features = 15;

	// allocate memory
	const int images_size = args.image_count * args.image_width *
		args.image_height * args.image_features;
	const int filters_size = args.output_features * args.image_features *
		args.filter_width * args.filter_height;
	const int outputs_size = args.image_count * args.image_width *
		args.image_height * args.output_features;
	args.images = new float[images_size];
	args.filters = new float[filters_size];
	args.outputs = new float[outputs_size];

	// initialize inputs
	for (int i = 0; i < images_size; i++)
		args.images[i] = generate_random_number();
	for (int i = 0; i < filters_size; i++)
		args.filters[i] = generate_random_number();

	convolution_cpu(args);

	// "read" outputs (for valgrind checking)
	volatile int sink;
	for (int i = 0; i < outputs_size; i++)
		sink = args.outputs[i];

	// free memory
	delete[] args.images;
	delete[] args.filters;
	delete[] args.outputs;

	return 0;
}

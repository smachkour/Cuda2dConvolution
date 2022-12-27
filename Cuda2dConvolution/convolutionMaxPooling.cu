#include <cuda.h>
#include <cuda_runtime.h>
//#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE 16


// convolution MaxPooling loop
__global__ void convMax(float* output, float* input, float* kernel, int input_width, int input_height)
{
    // 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the input image
    if (x < input_width && y < input_height)
    {
        // Compute the output pixel value
        float sum = 0;
        int tempMax = 0;
        int Max = 0;
        for (int ky = 0; ky < 2; ++ky)
        {
            for (int kx = 0; kx < 2; ++kx)
            {
                // Compute the index of the input pixel
                int px = x - 1 + kx;
                int py = y - 1 + ky;

                // Check if the pixel is within the bounds of the input image
                if (px >= 0 && px < input_width && py >= 0 && py < input_height)
                {
                    // Perform the convolution
                    sum += input[py * input_width + px] * kernel[ky * 2 + kx];
                    if (input[py * input_width + px] > tempMax)
                    {
                      tempMax = input[py * input_width + px];
                    }
                    max = tempMax;
                }
            }
        }
        // Set the output pixel value
        output[y * input_width + x] = max;
    }
}


int main()
{
    // Load the input image
    int width, height, componentCount;
    //cv::Mat input_image = cv::imread("puppyGrey.png", cv::IMREAD_GRAYSCALE);
     unsigned char *input_image = stbi_load("puppyGrey.png", &width, &height, &componentCount, 4);
    //if (input_image.empty())
    //{
    //    std::cerr << "Failed to load input image." << std::endl;
    //    return 1;
    //}

    // Get the image dimensions
    int input_width = width;
    int input_height = height;

    // Allocate host memory for the input image, kernel, and output image
    float* h_input = new float[input_width * input_height];
    float* h_kernel = new float[2 * 2];
    float* h_output = new float[input_width * input_height];

    // Copy the input image data to the host
    for (int y = 0; y < input_height; ++y)
    {
        for (int x = 0; x < input_width; ++x)
        {
            h_input[y * input_width + x] = static_cast<float>(input_image[(y * input_width + x) * 4]);
        }
    }

    // Set the values of the kernel
    h_kernel[0] = 1; h_kernel[1] = 1; h_kernel[2] = 1;
    h_kernel[3] = 1;// h_kernel[4] = 0; h_kernel[5] = 0;
 //   h_kernel[6] = 1; h_kernel[7] = 1; h_kernel[8] = 1;

// Allocate device memory
float* d_input;
float* d_kernel;
float* d_output;
int input_size = input_width * input_height;
int kernel_size = 3 * 3;
int output_size = input_width * input_height;
cudaMalloc((void**)&d_input, input_size * sizeof(float));
cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
cudaMalloc((void**)&d_output, output_size * sizeof(float));

// Copy data from host to device
cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

// Launch the kernel
dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
conv2D<<<grid, block>>>(d_output, d_input, d_kernel, input_width, input_height);

// Copy data from device to host
cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

// Create the output image
//cv::Mat output_image(input_height, input_width, CV_8UC1);
//for (int y = 0; y < input_height; ++y)
//{
//    for (int x = 0; x < input_width; ++x)
//    {
//        output_image.at<uchar>(y, x) = static_cast<uchar>(h_output[y * input_width + x]);
//    }
//}

unsigned char* h_output_image = (unsigned char*)malloc(input_size * sizeof(unsigned char));
for (int y = 0; y < input_height; ++y)
{
    for (int x = 0; x < input_width; ++x)
    {
        h_output_image[y * input_width + x] = static_cast<unsigned char>(h_output[y * input_width + x]);
    }
}

// Save the output image
// const char *fileNameOut = "blue.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
   // stbi_write_png(fileNameOut, width, height, 4, d_output, 4 * width);

    stbi_write_png("output_image.png", input_width, input_height, 1, h_output_image, input_width);


// Free host memory
delete[] h_input;
delete[] h_kernel;
delete[] h_output;

// Free device memory
cudaFree(d_input);
cudaFree(d_kernel);
cudaFree(d_output);

return 0;
}



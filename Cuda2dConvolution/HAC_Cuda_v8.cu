#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
	unsigned char r, g, b, a;
};

// *********************
// **** MIN POOLING ****
// *********************
__global__ void minPoolingGpu(unsigned char* imageRGBA, unsigned char* output, size_t stride, int width, int height)
{
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int x = 0; x < (height / 2); x++) // itterate height
	{
		for (int y = index; y < (width / 2); y += stride) // itterate width
		{
			// read pixel values
			Pixel* pix1 = (Pixel *) &imageRGBA[(y * 2 * 4) + (0 * 4) + (width * 0 * 4) + (x * 2 * width * 4)];
			Pixel* pix2 = (Pixel *) &imageRGBA[(y * 2 * 4) + (1 * 4) + (width * 0 * 4) + (x * 2 * width * 4)];

			Pixel* pix3 = (Pixel *) &imageRGBA[(y * 2 * 4) + (0 * 4) + (width * 1 * 4) + (x * 2 * width * 4)];
			Pixel* pix4 = (Pixel *) &imageRGBA[(y * 2 * 4) + (1 * 4) + (width * 1 * 4) + (x * 2 * width * 4)];

			// output pixel
			Pixel* ptrOutputPix = (Pixel *) &output[(y * 4) + (x * (width / 2) * 4)];
			// store value for output
			int tempPix;
            // RED channel
            tempPix = min(pix1->r, pix2->r); 
            tempPix = min(tempPix, pix3->r);
            tempPix = min(tempPix, pix4->r);
            ptrOutputPix->r = tempPix; 
            // GREEN channel
            tempPix = min(pix1->g, pix2->g); 
            tempPix = min(tempPix, pix3->g);
            tempPix = min(tempPix, pix4->g);
            ptrOutputPix->g = tempPix; 
            // BLUE channel
            tempPix = min(pix1->b, pix2->b); 
            tempPix = min(tempPix, pix3->b);
            tempPix = min(tempPix, pix4->b);
            if (DEBUG_PRINT)
                printf("Min (BLUE): %i\r\n", tempPix);
            ptrOutputPix->b = tempPix; 
            // ALPHA channel (transparantie)
            ptrOutputPix->a = 255; 
		}
	}
}

// *********************
// **** MAX POOLING ****
// *********************
__global__ void maxPoolingGpu(unsigned char* imageRGBA, unsigned char* output, size_t stride, int width, int height)
{
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int x = 0; x < (height / 2); x++) // itterate height
	{
		for (int y = index; y < (width / 2); y += stride) // itterate width
		{
			// read pixel values
			Pixel* pix1 = (Pixel *) &imageRGBA[(y * 2 * 4) + (0 * 4) + (width * 0 * 4) + (x * 2 * width * 4)];
			Pixel* pix2 = (Pixel *) &imageRGBA[(y * 2 * 4) + (1 * 4) + (width * 0 * 4) + (x * 2 * width * 4)];

			Pixel* pix3 = (Pixel *) &imageRGBA[(y * 2 * 4) + (0 * 4) + (width * 1 * 4) + (x * 2 * width * 4)];
			Pixel* pix4 = (Pixel *) &imageRGBA[(y * 2 * 4) + (1 * 4) + (width * 1 * 4) + (x * 2 * width * 4)];

			// output pixel
			Pixel* ptrOutputPix = (Pixel *) &output[(y * 4) + (x * (width / 2) * 4)];
			// store value for output
			int tempPix;
            // RED channel
            tempPix = max(pix1->r, pix2->r);
            tempPix = max(tempPix, pix3->r);
            tempPix = max(tempPix, pix4->r);
            if (DEBUG_PRINT)
                printf("Max (RED): %i\r\n", tempPix);
            ptrOutputPix->r = tempPix;
            // GREEN channel
            tempPix = max(pix1->g, pix2->g);
            tempPix = max(tempPix, pix3->g);
            tempPix = max(tempPix, pix4->g);
            if (DEBUG_PRINT)
                printf("Max (GREEN): %i\r\n", tempPix);
            ptrOutputPix->g = tempPix;
            // BLUE channel
            tempPix = max(pix1->b, pix2->b);
            tempPix = max(tempPix, pix3->b);
            tempPix = max(tempPix, pix4->b);
            if (DEBUG_PRINT)
                printf("Max (BLUE): %i\r\n", tempPix);
            ptrOutputPix->b = tempPix;
            // ALPHA channel (transparantie)
            ptrOutputPix->a = 255;

		}
	}
}

// *********************
// ***** GRAYSCALE *****
// *********************
__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA, size_t pixels, size_t stride, int width)
{
	int index = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 
	for (int i = index; i < pixels; i += stride)
    {
		Pixel* ptrPixel = (Pixel*)&imageRGBA[i]; 
		unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f); 
		ptrPixel->r = pixelValue; // Store the data directly into the input image
		ptrPixel->g = pixelValue;
		ptrPixel->b = pixelValue;
		ptrPixel->a = 255;
    }
}

// *********************
// **** CONVOLUTION ****
// *********************
__global__ void imageConvolutionGpu(unsigned char* inputImage, unsigned char* outputImage, size_t stride, int width, int height, int16_t * kernel)
{
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int x = 0; x < (height - 2); x++) // itterate height
	{
		for (int y = index; y < (width - 2); y += stride) // itterate width
		{
			// read matrix pixels values
			Pixel* pix1 = (Pixel *) &inputImage[(y * 4) + (0 * 4) + (width * 0 * 4) + (x * width * 4)];
			Pixel* pix2 = (Pixel *) &inputImage[(y * 4) + (1 * 4) + (width * 0 * 4) + (x * width * 4)];
			Pixel* pix3 = (Pixel *) &inputImage[(y * 4) + (2 * 4) + (width * 0 * 4) + (x * width * 4)];

			Pixel* pix4 = (Pixel *) &inputImage[(y * 4) + (0 * 4) + (width * 1 * 4) + (x * width * 4)];
			Pixel* pix5 = (Pixel *) &inputImage[(y * 4) + (1 * 4) + (width * 1 * 4) + (x * width * 4)];
			Pixel* pix6 = (Pixel *) &inputImage[(y * 4) + (2 * 4) + (width * 1 * 4) + (x * width * 4)];

			Pixel* pix7 = (Pixel *) &inputImage[(y * 4) + (0 * 4) + (width * 2 * 4) + (x * width * 4)];
			Pixel* pix8 = (Pixel *) &inputImage[(y * 4) + (1 * 4) + (width * 2 * 4) + (x * width * 4)];
			Pixel* pix9 = (Pixel *) &inputImage[(y * 4) + (2 * 4) + (width * 2 * 4) + (x * width * 4)];

			// output pixel
			Pixel* ptrOutputPix = (Pixel *) &outputImage[(y * 4) + (x * (width - 2) * 4)];
			// store value for output
			int ConValue;

			// RED channel
			ConValue = pix1->r * kernel[0] + pix2->r * kernel[1] + pix3->r * kernel[2] + pix4->r * kernel[3] + pix5->r * kernel[4] + pix6->r * kernel[5] + pix7->r * kernel[6] + pix8->r * kernel[7] + pix9->r * kernel[8];
			if (ConValue < 0){
                ptrOutputPix->r = 0; 
            }
			else if (ConValue > 255){
                ptrOutputPix->r = 255;
            } 
			else
				ptrOutputPix->r = ConValue; 
			// GREEN channel
			ConValue = pix1->g * kernel[0] + pix2->g * kernel[1] + pix3->g * kernel[2] + pix4->g * kernel[3] + pix5->g * kernel[4] + pix6->g * kernel[5] + pix7->g * kernel[6] + pix8->g * kernel[7] + pix9->g * kernel[8];
			if (ConValue < 0){
                ptrOutputPix->g = 0;
            }
			else if (ConValue > 255){
                ptrOutputPix->g = 255;
            }
			else
				ptrOutputPix->g = ConValue;
			// BLUE channel
			ConValue = pix1->b * kernel[0] + pix2->b * kernel[1] + pix3->b * kernel[2] + pix4->b * kernel[3] + pix5->b * kernel[4] + pix6->b * kernel[5] + pix7->b * kernel[6] + pix8->b * kernel[7] + pix9->b * kernel[8];
			if (ConValue < 0){
                ptrOutputPix->b = 0;
            }
			else if (ConValue > 255){
                ptrOutputPix->b = 255;
            }
			else
				ptrOutputPix->b = ConValue;
			// ALPHA channel
			ptrOutputPix->a = 255; 
		}
	}
}

// **********************
// **** MAIN PROGRAM ****
// **********************
int main(int argc, char** argv)
{
    // 3x3 KERNEL matrix
	int16_t kernel[9] = 
	{
		1, 0, -1,
		1, 0, -1,
		1, 0, -1
	};
    
    // preproces 
	char fileName[32];
	unsigned char *inputArr[10]; // input image data
	unsigned char *outputArr[3][10]; // output image data
    int componentCount;
	int width[10];
	int height[10];

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float GpuTimer; 

    printf("********************************\r\n");
    printf("** CONVOLUTION & POOLING TOOL **\r\n");
    printf("********************************\r\n\r\n");

	printf("reading images...\r\n");

	for (int i = 0; i < 10; i++) // Load all 10 images
	{
		memset(fileName, '\0', 32);
		char folder_path_input[100] = "/content/inputImg/";
		sprintf(fileName, "%sfoto%i.png", folder_path_input, i); // Create the file name, these are saves as: mage_in_x.png where x is a number from 0 to 9

		printf("loading : %s\r\n", fileName);

		inputArr[i] = stbi_load(fileName, &width[i], &height[i], &componentCount, 4);
		if (!inputArr[i])
            printf("failed to open image %i\r\n", i+(i*(-2)));

        // min pool
		outputArr[0][i] = (unsigned char *) malloc((width[i] / 2) * (height[i] / 2) * 4);
        // max pool
		outputArr[1][i] = (unsigned char *) malloc((width[i] / 2) * (height[i] / 2) * 4);
        // convolution
		outputArr[2][i] = (unsigned char *) malloc((width[i] - 2) * (height[i] - 2) * 4); 
	}

    // GPU
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	size_t RGBA[10]; 
	size_t threads_per_block[10];
	size_t number_of_blocks[10];
	size_t stride[10];

	int16_t *ptrKernel = nullptr;
	unsigned char *ptrInputImg[10];
	unsigned char *ptrOutputImg[3][10]; // min, avr, max and con

	for (int i = 0; i < 10; i++)
	{
		RGBA[i] = width[i] * height[i] * 4;
		threads_per_block[i] = min(properties.maxThreadsPerMultiProcessor, width[i]); 
		number_of_blocks[i] = (RGBA[i] + threads_per_block[i] - 1) / threads_per_block[i];
		stride[i] = threads_per_block[i] * number_of_blocks[i];
	}

	cudaEventRecord(start); 

	// Kernel alloc
	cudaMalloc(&ptrKernel, 9 * sizeof(int16_t)); 
	cudaMemcpy(ptrKernel, kernel, 9 * sizeof(int16_t), cudaMemcpyHostToDevice);

    // Host 2 Device
	for (int i = 0; i < 10; i++)
	{
		ptrInputImg[i] = nullptr;
		ptrOutputImg[0][i] = nullptr;
		ptrOutputImg[1][i] = nullptr;
		ptrOutputImg[2][i] = nullptr;

		cudaMalloc(&ptrInputImg[i], width[i] * height[i] * 4);
		cudaMalloc(&ptrOutputImg[0][i], (width[i] / 2) * (height[i] / 2) * 4);
		cudaMalloc(&ptrOutputImg[1][i], (width[i] / 2) * (height[i] / 2) * 4);
		cudaMalloc(&ptrOutputImg[2][i], (width[i] - 2) * (height[i] - 2) * 4);

		cudaMemcpy(ptrInputImg[i], inputArr[i], width[i] * height[i] * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(ptrOutputImg[0][i], outputArr[0][i], (width[i] / 2) * (height[i] / 2) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(ptrOutputImg[1][i], outputArr[1][i], (width[i] / 2) * (height[i] / 2) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(ptrOutputImg[2][i], outputArr[2][i], (width[i] - 2) * (height[i] - 2) * 4, cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();

	// min pool
	for (int i = 0; i < 10; i++)
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		minPoolingGpu<<<number_of_blocks[i], threads_per_block[i], 0, stream>>>(ptrInputImg[i], ptrOutputImg[0][i], stride[i], width[i], height[i]);
		cudaStreamDestroy(stream);
	}

	// max pool
	for (int i = 0; i < 10; i++)
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		maxPoolingGpu<<<number_of_blocks[i], threads_per_block[i], 0, stream>>>(ptrInputImg[i], ptrOutputImg[1][i], stride[i], width[i], height[i]);
		cudaStreamDestroy(stream);
	}
	cudaDeviceSynchronize(); 

	// Preprocess Convovlution
	for (int i = 0; i < 10; i++)
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		ConvertImageToGrayGpu<<<number_of_blocks[i], threads_per_block[i], 0, stream>>>(ptrInputImg[i], RGBA[i], stride[i], width[i]);
		cudaStreamDestroy(stream);
	}
	cudaDeviceSynchronize(); 

	// Convolution
	for (int i = 0; i < 10; i++)
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		imageConvolutionGpu<<<number_of_blocks[i], threads_per_block[i], 0, stream>>>(ptrInputImg[i], ptrOutputImg[2][i], stride[i], width[i], height[i], ptrKernel);
		cudaStreamDestroy(stream);
	}
	cudaDeviceSynchronize(); 

	// Device 2 Host
	for (int i = 0; i < 10; i++)
	{
		cudaMemcpy(outputArr[0][i], ptrOutputImg[0][i], (width[i] / 2) * (height[i] / 2) * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(outputArr[1][i], ptrOutputImg[1][i], (width[i] / 2) * (height[i] / 2) * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(outputArr[2][i], ptrOutputImg[2][i], (width[i] - 2) * (height[i] - 2) * 4, cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize(); 

	cudaEventRecord(stop); 
	cudaEventSynchronize(stop);
	GpuTimer = 0;
	cudaEventElapsedTime(&GpuTimer, start, stop);

	printf("writing files...\r\n");
	for (int i = 0; i < 10; i++)
	{
		memset(fileName, '\0', 16);
		sprintf(fileName, "/content/minImg/min_%i.png", i); 
		if (DEBUG_PRINT)
			printf("Image file name: %s", fileName);
		stbi_write_png(fileName, width[i] / 2, height[i] / 2, 4, outputArr[0][i], 4 * (width[i] / 2));

		memset(fileName, '\0', 16);
		sprintf(fileName, "/content/maxImg/max_%i.png", i);
		if (DEBUG_PRINT)
			printf("Image file name: %s", fileName);
		stbi_write_png(fileName, width[i] / 2, height[i] / 2, 4, outputArr[1][i], 4 * (width[i] / 2));

		memset(fileName, '\0', 16);
		sprintf(fileName, "/content/convImg/con_%i.png", i);
		if (DEBUG_PRINT)
			printf("Image file name: %s", fileName);
		stbi_write_png(fileName, width[i] - 2, height[i] - 2, 4, outputArr[2][i], 4 * (width[i] - 2));
	}

    // free memory
	cudaFree(ptrKernel);
	for (int i = 0; i < 10; i++)
	{
		cudaFree(ptrInputImg[i]);
		cudaFree(ptrOutputImg[0][i]);
		cudaFree(ptrOutputImg[1][i]);
		cudaFree(ptrOutputImg[2][i]);
		stbi_image_free(inputArr[i]);
		free(outputArr[0][i]);
		free(outputArr[1][i]);
		free(outputArr[2][i]);
	}

	printf("finished process...\r\n");

	printf("Process time on GPU : %3.9f ms\r\n", GpuTimer);

	return 0;
}
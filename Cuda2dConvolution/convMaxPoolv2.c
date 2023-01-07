#include <stdio.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char** argv) {

  // Load the input image using stb_image.h
  int width, height, num_channels;
  unsigned char* image = stbi_load("Before.png", &width, &height, &num_channels, 4);
  if (image == NULL) {
    fprintf(stderr, "Failed to load image %s\n", argv[1]);
    return 1;
  }

  // Create an output image with half the width and height of the input
  int output_width = width / 2;
  int output_height = height / 2;
  unsigned char* output_image = malloc(output_width * output_height * num_channels);

  // Perform max pooling on the input image and store the result in the output image
  for (int y = 0; y < output_height; y++) {
    for (int x = 0; x < output_width; x++) {
      for (int c = 0; c < num_channels; c++) {
        int max_value = 0;
        // Take the maximum value of the 2x2 group of pixels centered at (2x, 2y)
        for (int dy = 0; dy < 2; dy++) {
          for (int dx = 0; dx < 2; dx++) {
            int value = image[(2 * y + dy) * width + (2 * x + dx) * num_channels + c];
            if (value > max_value) {
              max_value = value;
            }
          }
        }
        output_image[y * output_width + x * num_channels + c] = max_value;
      }
    }
  }

  // Save the output image using stb_image.h
      // Write image back to disk
    printf("Writing png to disk...\r\n");

    stbi_write_png("output_image.png", output_width, output_height, num_channels, output_image, output_width);
  
  if (!stbi_write_png("output_image.png", output_width, output_height, num_channels, output_image, 0)) {
    fprintf(stderr, "Failed to save image %s\n", output_image);
    return 1;
  }

  // Clean up
  stbi_image_free(image);
  free(output_image);

  return 0;
}
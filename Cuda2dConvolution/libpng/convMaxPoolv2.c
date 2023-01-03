#include <stdio.h>
#include <stdlib.h>
#include "png.h"

#define WIDTH 800
#define HEIGHT 600
#define DEPTH 3
#define POOL_SIZE 2

int main() {
  // Load the image
  png_image image;
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_RGB;
  image.width = WIDTH;
  image.height = HEIGHT;
  png_image_begin_read_from_file(&image, "Before.png");
  png_bytep buffer = malloc(PNG_IMAGE_SIZE(image));
  png_image_finish_read(&image, NULL, buffer, 0, NULL);

  // Create the output image
  png_image output_image;
  output_image.version = PNG_IMAGE_VERSION;
  output_image.format = PNG_FORMAT_RGB;
  output_image.width = WIDTH / POOL_SIZE;
  output_image.height = HEIGHT / POOL_SIZE;
  png_bytep output_buffer = malloc(PNG_IMAGE_SIZE(output_image));

  // Perform the maxpooling
  for (int y = 0; y < HEIGHT; y += POOL_SIZE) {
    for (int x = 0; x < WIDTH; x += POOL_SIZE) {
      for (int d = 0; d < DEPTH; ++d) {
        // Find the maximum value in the pooling window
        int max_value = 0;
        for (int dy = 0; dy < POOL_SIZE; ++dy) {
          for (int dx = 0; dx < POOL_SIZE; ++dx) {
            int value = buffer[(y + dy) * WIDTH * DEPTH + (x + dx) * DEPTH + d];
            if (value > max_value) {
              max_value = value;
            }
          }
        }

        // Store the maximum value in the output image
        output_buffer[(y / POOL_SIZE) * output_image.width * DEPTH + (x / POOL_SIZE) * DEPTH + d] = max_value;
      }
    }
  }

  // Save the output image
  png_image_write_to_file(&output_image, "output.png", 0, output_buffer, 0, NULL);

  // Clean up
  free(buffer);
  free(output_buffer);
  png_image_free(&image);
  png_image_free(&output_image);

  return 0;
}
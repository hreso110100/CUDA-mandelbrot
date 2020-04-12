#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

void calculate(int height, int width, int max_iterations, double *red_pixels[800], double *green_pixels[800], double *blue_pixels[800]);

int main(int argc, char const *argv[]) {
  // Rozmery obrazka
  const int HEIGHT = 800;
  const int WIDTH = 800;
  // Maximalny pocet iteracii v ktorych bude prebiehat vypocet
  const int MAX_ITERATIONS = 100000;

  // CPU alokacia farebnych poli
  double *cpu_red_array[WIDTH];

  for (int i=0; i<WIDTH; i++) {
    cpu_red_array[i] = (double *)malloc(HEIGHT * sizeof(double));
  }

  double *cpu_green_array[WIDTH];

  for (int i=0; i<WIDTH; i++) {
    cpu_green_array[i] = (double *)malloc(HEIGHT * sizeof(double));
  }

  double *cpu_blue_array[WIDTH];

  for (int i=0; i<WIDTH; i++) {
    cpu_blue_array[i] = (double *)malloc(HEIGHT * sizeof(double));
  }

  // Zaciatok merania casu
  clock_t start = clock();

  // Vypocet
  calculate(HEIGHT, WIDTH, MAX_ITERATIONS, cpu_red_array, cpu_green_array, cpu_blue_array);

  // Koniec merania casu
  clock_t end = clock();
  double result_time = (double)(end - start) / CLOCKS_PER_SEC;

  printf("Vypocet trval: %.2f sekund.\n", result_time);

  // Zapis vysledkov do suboru
	FILE *file = fopen("output.txt", "w");

	for (int i = 0; i <WIDTH; i++) {
    for (int j = 0; j < HEIGHT; j++) {
		  fprintf(file, "%lf,%lf,%lf\n", cpu_red_array[i][j], cpu_green_array[i][j], cpu_blue_array[i][j]);
    }
	}

	fclose(file);

  // Uvolnenie zdrojov
  for(int i = 0; i < WIDTH; i++) {
    double* pointer = cpu_red_array[i];
    free(pointer);
  }

  for(int i = 0; i < WIDTH; i++) {
    double* pointer = cpu_green_array[i];
    free(pointer);
  }

  for(int i = 0; i < WIDTH; i++){
    double* pointer = cpu_blue_array[i];
    free(pointer);
  }

  return 0;
}

void calculate(int height, int width, int max_iterations, double *red_pixels[800], double *green_pixels[800], double *blue_pixels[800]) {
  // priblizenie mnoziny
  int zoom = 300;

	// Centrovanie obrazku
  double x_axis_offset = -(width)/1.4;
  double y_axis_offset = (height)/2.0;

  for (int y = 0; y < height; y++) {
    double c_img = (y_axis_offset - y)/zoom; // prevod suradnice y na komplexne cislo

    for (int x = 0; x < width; x++) {
      double c_real = (x_axis_offset + x)/zoom; // prevod suradnice x na komplexne cislo
      int iter = 0;
      double z_real = 0.0, z_imag = 0.0, z_real_tmp = 0.0, z_img_tmp = 0.0;
      double absolute = 0.0;

      // Funkcia na otestovanie ci cislo c patri do mnoziny --> z_n+1=z_n^2+c
      // Podmienka testu je |z| <=2
  		while (iter < max_iterations && absolute <= 4.0) {
  			z_real = z_real_tmp*z_real_tmp - z_img_tmp*z_img_tmp + c_real;
  			z_imag = 2.0*z_real_tmp*z_img_tmp + c_img;
  			absolute = z_real*z_real + z_imag*z_imag;

  			z_real_tmp = z_real;
  			z_img_tmp = z_imag;
  			iter++;
  		}

  		if (iter == max_iterations) {
  			// pixel bude zafarbeny na cierno
  			red_pixels[y][x] = 0.0;
  			green_pixels[y][x] = 0.0;
  			blue_pixels[y][x] = 0.0;
  		}
  		else {
  			// pixel bude zvyrazneny
  			red_pixels[y][x] = pow(((double) iter)/((double)max_iterations), 0.25);
  			green_pixels[y][x] = 1.0;

  			if (iter < max_iterations) {
  				blue_pixels[y][x] = 1.0;
  			}else {
  				blue_pixels[y][x] = 0.0;
  			}
  		}
    }
  }
}

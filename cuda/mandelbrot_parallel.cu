#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void calculate(int height, int width, int max_iterations, double *red_pixels, double *green_pixels, double *blue_pixels);

int main(int argc, char const *argv[]) {
	// Rozmery obrazka
	const int HEIGHT = 800;
	const int WIDTH = 800;
	// Maximalny pocet iteracii v ktorych bude prebiehat vypocet
	const int MAX_ITERATIONS = 100000;

	int total_pixels = HEIGHT*WIDTH;
	cudaEvent_t start_time, end_time;
	float result_time = 0;

	// CPU alokacia farebnych poli
	double * cpu_red_array = (double *) malloc(total_pixels * sizeof(double));
	double * cpu_green_array = (double *) malloc(total_pixels * sizeof(double));
	double * cpu_blue_array = (double *) malloc(total_pixels * sizeof(double));

	// GPU alokacia farebnych poli
	double *device_red_array, *device_green_array, *device_blue_array;

	cudaMalloc((void**)&device_red_array, total_pixels * sizeof(double));
	cudaMalloc((void**)&device_green_array, total_pixels * sizeof(double));
	cudaMalloc((void**)&device_blue_array, total_pixels * sizeof(double));

	// Zaciatok trvania vypoctu
	srand(time(NULL));
	cudaEventCreate(&start_time);
	cudaEventCreate(&end_time);
	cudaEventRecord(start_time, 0);

	// Vypocet na CUDE, grid ma 256 thread blokov, kde kazdy z nich ma 256 threadov
	calculate<<<256, 256>>>(HEIGHT, WIDTH, MAX_ITERATIONS, device_red_array, device_green_array, device_blue_array);

	// Koniec trvania vypoctu
	cudaEventRecord(end_time, 0);
	cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&result_time, start_time, end_time);

	printf("Vypocet trval: %.2f sekund.\n", result_time/1000);

	//Presun dat z device do CPU
	cudaMemcpy(cpu_red_array, device_red_array, total_pixels * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_green_array, device_green_array, total_pixels * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_blue_array, device_blue_array, total_pixels * sizeof(double), cudaMemcpyDeviceToHost);

	// Zapis vysledkov do suboru
	FILE *file = fopen("output.txt", "w");

	for (int i = 0; i <total_pixels; i++) {
		fprintf(file, "%lf,%lf,%lf\n", cpu_red_array[i], cpu_green_array[i], cpu_blue_array[i]);
	}

	fclose(file);

	// Uvolnenie zdrojov
	cudaFree(device_red_array);
	cudaFree(device_green_array);
	cudaFree(device_blue_array);
	free(cpu_red_array);
	free(cpu_green_array);
	free(cpu_blue_array);

	cudaEventDestroy(start_time);
	cudaEventDestroy(end_time);
	cudaDeviceReset();

	return 0;
}

__global__ void calculate(int height, int width, int max_iterations, double *red_pixels, double *green_pixels, double *blue_pixels) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; // Unikatny index CUDA threadu
	int total_pixels = height * width;
	int zoom = 300; // priblizenie mnoziny

	// Centrovanie obrazku
  double x_axis_offset = -(width)/1.4;
  double y_axis_offset = (height)/2.0;

	while (id < total_pixels) {
		// suradnice v obrazku
		int x = id / width;
		int y = id % height;

		int current_index = x + (y * width);

		double c_real = (x_axis_offset + x)/zoom; // prevod suradnice x na komplexne cislo
		double c_img = (y_axis_offset - y)/zoom; // prevod suradnice y na komplexne cislo
		double z_real = 0.0, z_imag = 0.0, z_real_tmp = 0.0, z_img_tmp = 0.0;
		double absolute = 0.0;

		int iter = 0;

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
			red_pixels[current_index] = 0.0;
			green_pixels[current_index] = 0.0;
			blue_pixels[current_index] = 0.0;
		}
		else {
			// pixel bude zvyrazneny
			red_pixels[current_index] = pow(((double) iter)/((double)max_iterations), 0.25);
			green_pixels[current_index] = 1.0;

			if (iter < max_iterations) {
				blue_pixels[current_index] = 1.0;
			}else {
				blue_pixels[current_index] = 0.0;
			}
		}
		id += blockDim.x * gridDim.x; //grid-type loop, specificke pre CUDU
	}
}

/*
 * Sorbonne Université  --  PPAR
 * Computing the Mandelbrot set, sequential version
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>               /* compile with -lm */
#include <sys/time.h>
#include <arpa/inet.h>          /* htonl */
#include <mpi.h>
#include "rasterfile.h"

char info[] = "\
Usage:\n\
      ./mandel dimx dimy xmin ymin xmax ymax depth\n\
\n\
      dimx,dimy : dimensions of the image to generate\n\
      xmin,ymin,xmax,ymax : domain to be computed, in the complex plane\n\
      depth : maximal number of iterations\n\
\n\
Some examples of execution:\n\
      ./mandel 3840 3840 0.35 0.355 0.353 0.358 200\n\
      ./mandel 3840 3840 -0.736 -0.184 -0.735 -0.183 500\n\
      ./mandel 3840 3840 -0.736 -0.184 -0.735 -0.183 300\n\
      ./mandel 3840 3840 -1.48478 0.00006 -1.48440 0.00044 100\n\
      ./mandel 3840 3840 -1.5 -0.1 -1.3 0.1 10000\n\
";

double wallclock_time()
{
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6);
}

unsigned char cos_component(int i, double freq)
{
	double iD = i;
	iD = cos(iD / 255.0 * 2 * M_PI * freq);
	iD += 1;
	iD *= 128;
	return iD;
}

/**
 *  Save the data array in rasterfile format
 */
void save_rasterfile(char *name, int largeur, int hauteur, unsigned char *p)
{
	FILE *fd = fopen(name, "w");
	if (fd == NULL) {
		perror("Error while opening output file. ");
		exit(1);
	}

	struct rasterfile file;
	file.ras_magic = htonl(RAS_MAGIC);
	file.ras_width = htonl(largeur);	            /* width of the image, in pixels */
	file.ras_height = htonl(hauteur);	            /* height of the image, in pixels */
	file.ras_depth = htonl(8);	                    /* depth of each pixel (1, 8 or 24 ) */
	file.ras_length = htonl(largeur * hauteur);	    /* size of the image in nb of bytes */
	file.ras_type = htonl(RT_STANDARD);	            /* file type */
	file.ras_maptype = htonl(RMT_EQUAL_RGB);
	file.ras_maplength = htonl(256 * 3);
	fwrite(&file, sizeof(struct rasterfile), 1, fd);

	/* Color palette: red component */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_component(i, 13.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	/* Color palette: green component */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_component(i, 5.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	/* Color palette: blue component */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_component(i + 10, 7.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	fwrite(p, largeur * hauteur, sizeof(unsigned char), fd);
	fclose(fd);
}

/**
 * Given the coordinates of a point $c = a + ib$ in the complex plane,
 * the function returns the corresponding color which estimates the
 * distance from the Mandelbrot set to this point.
 * Consider the complex sequence defined by:
 * \begin{align}
 *     z_0     &= 0 \\
 *     z_{n+1} &= z_n^2 + c
 *   \end{align}
 * The number of iterations that this sequence needs before diverging
 * is the number $n$ for which $|z_n| > 2$.
 * This number is brought back to a value between 0 and 255, thus
 * corresponding to a color in the color palette.
 */
unsigned char xy2color(double a, double b, int depth)
{
	double x = 0;
	double y = 0;
	for (int i = 0; i < depth; i++) {
		/* save the former value of x (which will be erased) */
		double temp = x;
		/* new values for x and y */
		double x2 = x * x;
		double y2 = y * y;
		x = x2 - y2 + a;
		y = 2 * temp * y + b;
		if (x2 + y2 > 4.0)
			return (i % 255);   /* diverges */
	}
	return 255;   /* did not diverge in depth steps */
}

/* 
 * Main: in each point of the grid, apply xy2color
 */
int main(int argc, char **argv)
{
	if (argc == 1) {
		fprintf(stderr, "%s\n", info);
		return 1;
	}

	/* Default values for the fractal */
	double xmin = -2;     /* Domain of computation, in the complex plane */
	double ymin = -2;
	double xmax = 2;
	double ymax = 2;
	int w = 3840;         /* dimensions of the image (4K HTDV!) */
	int h = 3840;
	int depth = 10000;     /* depth of iteration */

	/* Retrieving parameters */
	if (argc > 1)
		w = atoi(argv[1]);
	if (argc > 2)
		h = atoi(argv[2]);
	if (argc > 3)
		xmin = atof(argv[3]);
	if (argc > 4)
		ymin = atof(argv[4]);
	if (argc > 5)
		xmax = atof(argv[5]);
	if (argc > 6)
		ymax = atof(argv[6]);
	if (argc > 7)
		depth = atoi(argv[7]);


	/* Computing steps for increments */
	double xinc = (xmax - xmin) / (w - 1);
	double yinc = (ymax - ymin) / (h - 1);

	/* show parameters, for verification */
	fprintf(stderr, "Domaine: [%g,%g] x [%g,%g]\n", xmin, ymin, xmax, ymax);
	fprintf(stderr, "Increment : %g, %g\n", xinc, yinc);
	fprintf(stderr, "depth: %d\n", depth);
	fprintf(stderr, "Dim image: %d x %d\n", w, h);

    int rank, num_procs;
    unsigned char *image, *local_image;
    int rows_per_proc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    rows_per_proc = h / num_procs;
    int local_start = rank * rows_per_proc;
    int local_end = local_start + rows_per_proc;
    local_image = malloc(w * rows_per_proc * sizeof(unsigned char)); // *3 for RGB channels

	/* Allocating memory for the output array */
	if (rank == 0)
	{
		image = malloc(w * h);
		if (image == NULL) {
			perror("Error while allocating memory for array: ");
			exit(1);
		}
	}

	/* starting timer */
	double start = wallclock_time();

	/* Processing the grid point by point */
	double y = ymin + local_start * yinc;
	for (int i = local_start; i < local_end; i++) {
		double x = xmin;
		for (int j = 0; j < w; j++) {
			local_image[j + (i - local_start)* w] = xy2color(x, y, depth);
			x += xinc;
		}
		y += yinc;
	}

	MPI_Gather(local_image, w * rows_per_proc, MPI_UNSIGNED_CHAR, image, w * rows_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	/* ending timer */
	double end = wallclock_time();
	fprintf(stderr, "Total computing time: %g sec\n", end - start);

	if (rank == 0)
	{
		/* Saving the image in the output file "mandel.ras" */

		save_rasterfile("mandel_parallel.ras", w, h, image);
        fprintf(stderr, "Image saved in mandel_parallel.ras with time : %g sec\n", end-start);

		/* Freeing memory */
		free(image);
	}
	free(local_image);

	MPI_Finalize();
	return 0;
}

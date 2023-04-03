
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

inline unsigned char iter(double a, double b);
void makefractal_cpu(unsigned char *image, int width, int height, double xupper, double xlower, double yupper, double ylower);
void save_image(unsigned char *image, int width, int height, char *filename);

int main(int argc, char **argv) 
{
//  double xupper=-0.12954, xlower=-0.1992, yupper=1.06707, ylower=1.0148;
//  double xupper=-0.88333, xlower=-0.95, yupper=0.3, ylower=0.23333;
//  double xupper=-0.4082, xlower=-0.73, yupper=0.71429, ylower=0.49216;
//  double xupper=-1.764, xlower=-1.781, yupper=0.013, ylower=0;
//  double xupper=-0.7408, xlower=-0.75104, yupper=0.11536, ylower=0.10511;
//  double xupper=-0.74624, xlower=-0.74758, yupper=0.10779, ylower=0.10671;  // try this
//  double xupper=-0.745054, xlower=-0.745538, yupper=0.113236, ylower=0.112881;
//  double xupper=-0.745385, xlower=-0.745468, yupper=0.113039, ylower=0.112979;
//  double xupper=-0.7454215, xlower=-0.7454356, yupper=0.1130139, ylower=0.1130037;
//  double xupper=-1.252861, xlower=-1.254024, yupper=0.047125, ylower=0.046252;
//  double xupper=1.000000000000, xlower=-2.100000000000, yupper=1.300000000000, ylower=-1.300000000000;
  double xupper=-0.754534912109, xlower=-0.757077407837, yupper=0.060144042969, ylower=0.057710774740;   // try this
//  double xupper=-0.754899899289, xlower=-0.755048873648, yupper=0.059548399183, ylower=0.059418498145;
//  double xupper=-0.755030251853, xlower=-0.755038398888, yupper=0.059496303455, ylower=0.059491229195;
//  double xupper=-0.205592346191, xlower=-0.220729064941, yupper=-0.679331461589, ylower=-0.690483940972;
//  double xupper=-0.212894630432, xlower=-0.214180660248, yupper=-0.685619252699, ylower=-0.686737404929;
//  double xupper=0.349307936343, xlower=0.349306702708, yupper=-0.584894511814, ylower=-0.584895037624;

    int width = 1024;
    int height = 768;
    unsigned char *image = NULL;

    struct timeval t1, t2;
    long msec1, msec2;

    image = (unsigned char*)malloc(width*height*sizeof(unsigned char*));

    gettimeofday(&t1, NULL);
    msec1 = t1.tv_sec * 1000000 + t1.tv_usec;

    makefractal_cpu(image, width, height, xupper, xlower, yupper, ylower);
    
    gettimeofday(&t2, NULL);
    msec2 = t2.tv_sec * 1000000 + t2.tv_usec;

    printf("Generated fractal in %.2f seconds\n", (float)(msec2-msec1)/1000000.0f);

    save_image(image, width, height, "fractal.ppm");

    free(image);
}


// ================= CALC

void makefractal_cpu(unsigned char *image, int width, int height, double xupper, double xlower, double yupper, double ylower)
{
    double yinc;
    double xinc;
    int x, y;

    xinc   = (xupper - xlower) / width;
    yinc   = (yupper - ylower) / height;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            image[y*width+x] = iter((xlower + x*xinc), (ylower + y*yinc));
        }
    }
}


inline unsigned char iter(double a, double b)
{
    unsigned char i = 0;
    double c_x = 0, c_y = 0;
    double c_x_tmp, c_y_tmp;
    double D = 4.0;

    while ((c_x*c_x+c_y*c_y < D) && (i++ < 255))
    {
        c_x_tmp = c_x * c_x - c_y * c_y;
        c_y_tmp = 2* c_y * c_x;

        c_x = a + c_x_tmp;
        c_y = b + c_y_tmp;
    }

    return i;
}

void save_image(unsigned char *image, int width, int height, char *filename)
{
    FILE *fd = NULL;
    int i, x, y;
    struct { int r, g, b; } colors[256];
    
    for (i = 0; i < 256 ; i++) 
    {
        colors[i].r = abs(((i + 60) % 256) - 127);
        colors[i].g = abs(((i + 160) % 256) - 127);
        colors[i].b = abs(((i + 200) % 256) - 127);
    }
   
    
    fd = fopen(filename, "w");
    
    fprintf(fd, "P3\n%d %d\n255\n", width, height);
    
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
	    int pixel = image[y*width+x];
            fprintf(fd, "%i %i %i\n", colors[pixel].r, colors[pixel].g, colors[pixel].b);
        }
    }
    
    fclose(fd);

}




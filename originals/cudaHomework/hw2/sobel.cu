

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include "cudaSobel.cu"
#include <helper_image.h>

#define DEFAULT_THRESHOLD  4000

#define DEFAULT_FILENAME "BWstop-sign.ppm"




unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  FILE *fp;

  fprintf(stderr, "read_ppm( %s )\n", filename);
  fp = fopen( filename, "rb");
  if (!fp) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 

    }

  char chars[1024];
  //int num = read(fd, chars, 1000);
  int num = fread(chars, sizeof(char), 1000, fp);

  if (chars[0] != 'P' || chars[1] != '6') 
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line! 
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
  *xsize = width;
  *ysize = height;
  *maxval = maxvalue;
  
  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if ((*maxval) > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }





  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", *xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", *ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", *maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  //lseek(fd, offset, SEEK_SET); // move to the correct offset
  fseek(fp, offset, SEEK_SET); // move to the correct offset
  //long numread = read(fd, buf, bufsize);
  long numread = fread(buf, sizeof(char), bufsize, fp);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

  fclose(fp);


  int pixels = (*xsize) * (*ysize);
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

 

  return pic; // success
}












void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
  FILE *fp;
  int x,y;
  
  fp = fopen(filename, "w");
  if (!fp) 
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
      exit(-1); 
    }
  
  
  
  fprintf(fp, "P6\n"); 
  fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
  int numpix = xsize * ysize;
  for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc); 
  }
  fclose(fp);

}

int* standardSobel(int *result, unsigned int *pic, int xsize, int ysize, int thresh)
{
  int i, j, magnitude, sum1, sum2; 
  int *out = result;

  for (int col=0; col<ysize; col++) {
    for (int row=0; row<xsize; row++) { 
      *out++ = 0; 
    }
  }

  for (i = 1;  i < ysize - 1; i++) {
    for (j = 1; j < xsize -1; j++) {
      
      int offset = i*xsize + j;

      sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
        + 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
        +     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];
      
      sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
            - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];
      
      magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > thresh)
        result[offset] = 255;
      else 
        result[offset] = 0;
    }
  }
  return result;
}

int compareNumPixels(int *result1, int *result2, int xsize, int ysize)
{
	int numPixels = xsize * ysize;
	for (int i = 0; i<numPixels; i++)
	{
		if((unsigned char)result1[i] != (unsigned char)result2[i])
		{
			printf("pixel at i= %d,  not equal 1 = %d, 2= %d, \n", i, result1[i], result2[i]);
			return -1; 

		}
	}
	return 1;
}

int compare(int *result1, int * result2, int xsize, int ysize)
{
	printf("starting compare \n");
	for (int i = 0; i<ysize; i ++)
	{
		for (int j = 0; j<xsize; j++)
		{
			int index = i*xsize+j;
			if ((unsigned char)result1[index] != (unsigned char)result2[index])
			{
				printf("xsize = %d, ysize = %d, pixel at I= %d, j= %d, index %d,  not equal 1 = %d, 2= %d, \n", xsize, ysize, i, j, index, result1[index], result2[index]);
				compareNumPixels(result1, result2, xsize, ysize);
				exit(-1); 
			}
		}
	}
	return 1;
}

int main( int argc, char **argv )
{

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  filename = strdup( DEFAULT_FILENAME);
  
  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
  }


  int xsize, ysize, maxval;
  unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval ); 


  int numbytes =  xsize * ysize * 3 * sizeof( int );
  //printf("numbytes = %d \n", numbytes);
  int *result = (int *) malloc( numbytes );
  int *gpu1Result = (int *) malloc(numbytes);
  int *gpu2Result = (int *) malloc(numbytes);

  if (!result) { 
    fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
    exit(-1); // fail
  }
  if(!gpu1Result) {
    fprintf(stderr, "gpu1result unable to malloc %d bytes \n", numbytes);
    exit(-1); // failed again
  }
  if(!gpu2Result) {
    fprintf(stderr, "gpu2result unable to malloc %d bytes \n", numbytes);
    exit(-1); // failed again
  }

  printf("single threaded sobel\n");
  // non threaded sobel

  //setup some timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  result = standardSobel(result, pic, xsize, ysize, thresh);

  // finish some timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Single threaded elapsed time, kernel only = %f \n", elapsedTime);

  write_ppm( "result.ppm", xsize, ysize, 255, result);
  
  printf("gpu1 \n");
  sobel1(gpu1Result, pic, xsize, ysize, thresh);
  write_ppm( "gpu1Result.ppm", xsize, ysize, 255, gpu1Result);

  printf("gpu2 \n");
  sobel2(gpu2Result, pic, xsize, ysize, thresh);
  write_ppm( "gpu2Result.ppm", xsize, ysize, 255, gpu2Result);
  fprintf(stderr, "sobel done\n"); 



  // compare the single threaded and multithreaded results
  if(compareData(result, gpu1Result, xsize*ysize, 0, 0.0f) == false);
	  compare(result, gpu1Result, xsize, ysize);
  
  if(compareData(result, gpu2Result, xsize*ysize, 0, 0.0f) == false);
	  compare(result, gpu2Result, xsize, ysize);

}


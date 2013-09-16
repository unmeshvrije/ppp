#include <stdio.h>
#include <iostream>
#include <cstring>

int PadImage(
	unsigned char *image,
	int width,
	int height,
	int factor,
	unsigned char **pNewImage
	)
{
  // 
  // Assumes that padding is to be done. i.e. width and wantedWidth ARE DIFFERENT
  //
  unsigned char *newImage;
  int n = width / factor; // 1302 / 16

  int newWidth = (n+1) * factor;

  newImage = new unsigned char[newWidth * height];

  unsigned char *zeroBuffer = new unsigned char[newWidth - width];

  memset(zeroBuffer, 0, (newWidth-width) * sizeof(unsigned char));

  for (int i = 0; i < height; ++i)
  {
     memcpy(newImage + (i * newWidth)	     , image + (i * width), width * sizeof(unsigned char));
     memcpy(newImage + (i * newWidth) + width, zeroBuffer, (newWidth - width) * sizeof(unsigned char));

//     printf("Offset of valid = %d\n", i * newWidth);
//     printf("Offset of zeros = %d\n", i * newWidth + width);
  }

  delete[] zeroBuffer;

  *pNewImage = newImage;
  return newWidth;
}


int Unpad(
	unsigned char *image,
	int newWidth,
	int width,//original width only
	int height,
	unsigned char *newImage // already allocated
	)
{
  for (int i = 0; i < height; ++i)
  {
    memcpy(newImage + (i * width), image + (i * newWidth), width *(sizeof(unsigned char)));
  }
}

void
PrintImage(unsigned char *image, int width, int height)
{
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      printf("%2d ", image[ i * width + j]);
    }
    printf("\n");
  }
}



int main()
{
   unsigned char *newImage = NULL;
   unsigned char image [20] = {
   				1,1,1,1,1,
   				1,2,1,1,1,
   				1,1,3,1,1,
   				1,1,1,4,1,
   				};

  int width = 5;
  int height = 4;
  PrintImage(image, width,height);

  int newWidth= PadImage(image, width,height,8, &newImage);

  printf("\n\n");
  PrintImage(newImage, newWidth,4);

  printf("Unpad again\n\n");
  Unpad(newImage, newWidth, width, height, image);
  PrintImage(image,width,height);

  delete[] newImage;
}

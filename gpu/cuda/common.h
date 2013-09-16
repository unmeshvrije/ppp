#define	ARRAY_SIZE(X)	(sizeof(X) / (sizeof((X)[0])))

const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;

// Filter is 5 X 5
#define	FILTER_WIDTH	5
#define	TILE_WIDTH 8


//
// Returns newWidth of image
//
int PadImage(
	unsigned char *image,
	int width,
	int height,
	int factor,// 16 if you want width to be multiple of 16
	unsigned char **newImage
	);

void UnpadImage(
	unsigned char *BigImage,
	int BigWidth,
	int width,
	int height,
	unsigned char *OrigImage
	);

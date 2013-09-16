#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <limits.h>
#include <sys/time.h>

#include "../common/common.h"

#define	MAX_PATH	256

#define	MIN(X,Y)	((X < Y) ? (X) : (Y))
#define	MAX(X,Y)	((X > Y) ? (X) : (Y))

#define	OUTPUT_FILE	"seqmat.out"
#define	PATH_OUTPUT_FILE	"seqpath.out"

int ParseFile(const char *szFileName, int *pN, int *pE, int ***pppAdjMat, int ***pppNext)
{
	int i,j;
	int n, e;
	FILE *fp;
	int **graph;
	int **next;
	int orientation;
	int src, dest, weight;

	if (NULL == szFileName || NULL == pN || NULL == pE || NULL == pppAdjMat)
		return -1;

	fp = fopen(szFileName ,"r");
	if (NULL == fp)
	{
		//log
		return -1;
	}

	fscanf(fp, "%d %d %d", &n, &e, &orientation);

	// Allocate memory
	// +

	graph = (int**)malloc(sizeof(int*) * n);
	next = (int**)malloc(sizeof(int*) * n);
	if (NULL == graph || NULL == next)
	{
		printf("FATAL: out of memory\n");
		return -1;
	}

	for (i = 0; i < n; ++i)
	{
		graph[i] = (int*)calloc(n, sizeof(int));
		next[i] = (int*)calloc(n, sizeof(int));
		if (NULL == graph[i] || NULL == next[i])
		{
			printf("FATAL: out of memory\n");
			return -1;
		}

		for (j = 0; j < n; ++j)
			if (i != j)
				graph[i][j] = INFINITY;
	}
	//-
	// Allocate memory

	for (i = 0; i < e; ++i)
	{
		fscanf(fp, "%d %d %d", &src, &dest, &weight);
		graph[src-1][dest-1] = weight;
		if (0 == orientation)
		{
			graph[dest-1][src-1] = weight;
		}
	}


	fclose(fp);

	*pN = n;
	*pE = e;
	*pppAdjMat = graph;
	*pppNext = next;

	return 0;
}

void PrintAdjMat(int **ppAdjMat, int N)
{
	int i,j;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
			printf("%15d" , ppAdjMat[i][j]);

		printf("\n");
	}
}

void asp(int **ppAdjMat, int **ppNext, int n)
{
	int i,j,k;

	for (k = 0; k < n; ++k)
	{
		for (i = 0; i < n; ++i)
		{
			if (i != k)
			{
				for (j = 0; j < n; ++j)
				{
					if (ppAdjMat[i][k] + ppAdjMat[k][j] < ppAdjMat[i][j])
					{
					  ppAdjMat[i][j] = ppAdjMat[i][k] + ppAdjMat[k][j];
					  ppNext[i][j] = k;
					}
				}
			}
		}
	}
}

// n is number of rows
void FreeGraph(int **ppAdjMat, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		free (ppAdjMat[i]);
	}

	free (ppAdjMat);
}

int calculate_total_road_distance(int **ppAdjMat, int M, int N)
{
	int i,j;
	int d = 0;
	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
			if (INFINITY != ppAdjMat[i][j])
				d += ppAdjMat[i][j];
	}

	return d;
}

int calculate_diameter(int **ppAdjMat, int M, int N)
{
	int i,j;
	int d = 0;
	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
			if (INFINITY != ppAdjMat[i][j] && ppAdjMat[i][j] > d)
				d = ppAdjMat[i][j];
	}

	return d;
}

int main(int argc, char *argv[])
{
	int N,E;
	int **ppAdjMat;
	int **ppNext;
	char szFileName[MAX_PATH];

	int totalDistance;
	int diameter;

	double elapsedTime;
	struct timezone tz;
	struct timeval End;
	struct timeval Start;

	int PrintPath = 0;

	if (argc < 0 || argc > 3)
	{
		printf("Usage: asp <path of road_list> [p]\n");
		printf("p : Paths will be generated in file\n");
		return 0;
	}

	if (strlen(argv[1]) >= MAX_PATH)
	{
		printf("File name too long.\n");
		return 1;
	}

	if (argc > 2)
	{
	  if (argv[2][0] == 'p' || argv[2][0] == 'P')
	  {
	    PrintPath = 1;
	  }
	  else
	  {
		printf("Invalid option\n");
		return 0;
	  }
	}

	strcpy(szFileName, argv[1]);
	N = 0;
	E = 0;
	ppAdjMat = NULL;
	ParseFile(szFileName, &N, &E, &ppAdjMat, &ppNext);

	memset(&Start, 0L, sizeof(struct timeval));
	memset(&End, 0L, sizeof(struct timeval));
	memset(&tz, 0L, sizeof(struct timezone));
	gettimeofday(&Start, &tz);

	totalDistance =	calculate_total_road_distance(ppAdjMat, N, N);
	asp(ppAdjMat, ppNext,N);
	diameter = calculate_diameter(ppAdjMat, N, N);

	gettimeofday(&End, &tz);

	elapsedTime = (End.tv_sec - Start.tv_sec) * 1000.0;
	elapsedTime += (End.tv_usec - Start.tv_usec) / 1000.0;

	printf("File name:<%s>\n", szFileName);
	printf("Number of vertices = %d\n", N);
	printf("Total road distance = %d\n", totalDistance/2);
	printf("Diameter = %d\n", diameter);
	printf("%s %f milliseconds\n", "Total time = ", elapsedTime);

	PrintAdjMatToFile(OUTPUT_FILE, ppAdjMat, N, N);
	if (PrintPath == 1)
	{
	  PrintPathsToFile(PATH_OUTPUT_FILE, ppAdjMat, ppNext, N, N);
	}

	FreeGraph(ppAdjMat, N);
	return 0;
}


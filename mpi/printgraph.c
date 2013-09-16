#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <limits.h>

#define	MAX_PATH	256
#define INFINITY	10000//INT_MAX

#define	MIN(X,Y)	((X < Y) ? (X) : (Y))
#define	MAX(X,Y)	((X > Y) ? (X) : (Y))

#define	OUTPUT_FILE	"seqmat.out"

int ParseFile(const char *szFileName, int *pN, int *pE, int ***pppAdjMat)
{
  int i,j;
  int n, e;
  FILE *fp;
  int **graph;
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
  if (NULL == graph)
  {
    printf("FATAL: out of memory\n");
    return -1;
  }

  for (i = 0; i < n; ++i)
  {
    graph[i] = (int*)calloc(n, sizeof(int));
    if (NULL == graph[i])
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

int main(int argc, char *argv[])
{
  int N,E;
  int **ppAdjMat;
  char szFileName[MAX_PATH];

  if (2 != argc)
  {
    printf("Usage: asp <path of road_list>\n");
    return 0;
  }

  if (strlen(argv[1]) >= MAX_PATH)
  {
    printf("File name too long.\n");
    return 1;
  }

  strcpy(szFileName, argv[1]);
  N = 0;
  E = 0;
  ppAdjMat = NULL;
  ParseFile(szFileName, &N, &E, &ppAdjMat);
  PrintAdjMat(ppAdjMat, N);
  FreeGraph(ppAdjMat, N);
  return 0;
}


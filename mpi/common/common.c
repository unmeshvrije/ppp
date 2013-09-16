#include "common.h"

#include <stdarg.h>

void
WriteToLog(
		const char *pcszFilePath, 
		const char *pcszMessage
	  )
{
	int	ifd;
	int iRet;

	if (NULL == pcszFilePath || NULL == pcszMessage)
	{
		return;
	}

	iRet = access(pcszFilePath, 00);

	if (0 == iRet)
	{
		ifd = open(pcszFilePath, O_RDWR | O_APPEND);	
	}
	else
	{
		ifd = open(pcszFilePath, O_RDWR | O_CREAT | O_TRUNC, S_IWRITE | S_IREAD);
	}

	if(-1 == ifd)
	{
		return;
	}

	iRet = write(ifd, pcszMessage, (unsigned int)strlen(pcszMessage) * sizeof(char));
	iRet = write(ifd, ("\r\n"), (unsigned int)strlen(("\r\n")) * sizeof(char));

	close(ifd);
}

void
WriteToLogf(
		const char *pcszFilePath, 
		const char *pcszFormat,
		...
	  )
{
	int ifd;
	int iRet;
	va_list arg_ptr;
	char szBuffer[1024];

	if (NULL == pcszFilePath || NULL == pcszFormat)
	{
		return;
	}

	iRet = access(pcszFilePath, 00);

	if (0 == iRet)
	{
		ifd = open(pcszFilePath, O_RDWR | O_APPEND);	
	}
	else
	{
		ifd = open(pcszFilePath, O_RDWR | O_CREAT | O_TRUNC, S_IWRITE | S_IREAD);
		//ifd = open(pcszFilePath, O_RDWR | O_CREAT | O_TRUNC, S_IREAD);
	}

	if(-1 == ifd)
	{
		return;
	}

	va_start(arg_ptr, pcszFormat);
	iRet = vsnprintf(szBuffer, ARRAY_SIZE(szBuffer) - 1, pcszFormat, arg_ptr);
	va_end(arg_ptr);

	if (-1 == iRet)
	{
	  return;
	}

	lseek(ifd, 0, SEEK_END);	
	iRet = write(ifd, szBuffer, ((unsigned int)strlen(szBuffer) ) * sizeof(char));

	iRet = write(ifd, ("\r\n"), (unsigned int)strlen(("\r\n")) * sizeof(char));

	close(ifd);
}

void PrintAdjMatToFile(const char *szOutputFile, int **ppAdjMat, int M, int N)
{
  int i,j;
  FILE *fp = fopen(szOutputFile, "w");
  if (NULL == fp)
    return;

  for (i = 0; i < M; ++i)
  {
    for (j = 0; j < N; ++j)
      fprintf(fp, "%15d" , ppAdjMat[i][j]);

    fprintf(fp, "\n");
  }
}

void
Path(int **ppDist, int **ppNext, int i, int j, char *szPath)
{
  char szPath1[MAX_PATH_STR_LEN];
  char szPath2[MAX_PATH_STR_LEN];
  char szInter[15];
  int intermediate;
  if (ppDist[i][j] == INFINITY)
  {
    strcpy(szPath, "no path");
    return;
  }

  intermediate = ppNext[i][j];
  if (0 == intermediate)
  {
    strcpy(szPath,"");
    return;
  }

  Path(ppDist, ppNext, i, intermediate, szPath1); 
  Path(ppDist, ppNext, intermediate, j, szPath2);

  strcpy(szPath, szPath1);
  sprintf(szInter,"%d->", intermediate);
  strcat(szPath, szInter);
  strcat(szPath, szPath2);  
}


void 
PrintPathsToFile(const char *szOutputFile, int **ppAdjMat, int **ppNext,int M, int N)
{
  int i,j;
  char szPath[MAX_PATH_STR_LEN];
  FILE *fp = fopen(szOutputFile, "w");
  if (NULL == fp)
    return;

  for (i = 0; i < M; ++i)
  {
    for (j = 0; j < N; ++j)
    {
      if (i == j)
        continue;

      memset(szPath, 0, ARRAY_SIZE(szPath));
      Path(ppAdjMat, ppNext, i, j, szPath);
      fprintf(fp, "<%d>to<%d> : %4d : %d->%s%d\n", i, j, ppAdjMat[i][j], i, szPath, j);
    }
  }
}

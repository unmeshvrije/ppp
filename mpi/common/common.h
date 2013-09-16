#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifndef _MYCOMMON_H
#define _MYCOMMON_H


typedef unsigned char BYTE;

#define	TRUE	1
#define	FALSE	0

#define	INFINITY	10000
#define	MAX_PATH_STR_LEN	4096
#define	ARRAY_SIZE(X)	((sizeof(X)) / (sizeof((X)[0])))

void
WriteToLog(
		const char *pcszFilePath, 
		const char *pcszMessage
	  );

void
WriteToLogf(
		const char *pcszFilePath, 
		const char *pcszMessage,
		...
	  );
void
PrintAdjMatToFile(
  const char *szFile,
  int **ppMat,
  int M,
  int N
  );

void
PrintPathsToFile(
  const char *szFile,
  int **ppMat,
  int **ppNext,
  int M,
  int N
  );

#endif//_MYCOMMON_H

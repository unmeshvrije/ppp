#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <limits.h>

#include <mpi.h>
#include "../common/common.h"

#include <unistd.h>	//access

#define	MAX_PATH	256

#define	MIN(X,Y)	((X < Y) ? (X) : (Y))
#define	MAX(X,Y)	((X > Y) ? (X) : (Y))


#define	LOG_FILE	"log.dat"
#define	OUTPUT_FILE	"adjmat.out"
#define	PATH_OUTPUT_FILE	"paths.out"

#define	DEBUG_LOG 0
#define	PERF_CONSOLE 1

double CommunicationTime = 0.0;
double ComputationTime = 0.0;
double StartTime = 0.0;
double EndTime = 0.0;
double globalCommunicationTime = 0.0;
double globalComputationTime = 0.0;

int malloc_int_2d(int ***pppMat, int m, int n)
{
	int i;

	//
	// Actually we want m * n integers
	// 
	int *p = (int*) calloc(m * n, sizeof (int));
	if (NULL == p)
		return -1;

	//
	// Allocate memory for m (int*)s
	//
	//   	|o|->
	//	|o|->
	//	|o|->
	//
	*pppMat = (int**)malloc(sizeof(int*) * m);
	if (NULL == pppMat)
	{
		free (p);
		return -1;
	}

	for (i = 0; i < m; ++i)
		(*pppMat)[i] = &(p[i * n]);//Brackets are crucial

	return 0;
}


void
free_int_2d(int ***p)
{
	free (&((*p)[0][0])); // Free 1D array

	free (*p);// Free 2D array
}

int
ParseFile(const char *szFileName, int *pN, int *pE, int ***pppAdjMat, int ***pppNext)
{
	int i, j;
	int n, e;
	int iRet;
	FILE *fp;
	int **graph = NULL;
	int **next = NULL;
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

	iRet = malloc_int_2d(&graph, n,n);
	if (-1 == iRet)
	{
		return -1;
	}

	iRet = malloc_int_2d(&next, n, n);
	if (-1 == iRet)
	{
		return -1;
	}

	for(i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			if (i != j)
				graph[i][j] = INFINITY;

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

void PrintAdjMat(int **ppAdjMat, int M, int N)
{
	int i,j;
	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
			printf("%15d" , ppAdjMat[i][j]);

		printf("\n");
	}
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

void asp(int **ppAdjMat, int **ppNext, int n)
{
	int i,j,k;
	for (k = 1; k < n; ++k)
	{
		for (i = 0; i < n; ++i)
		{
			if (i != k)
			{
				for (j = 0; j < n; ++j)
				{
					if ( ppAdjMat[i][k] + ppAdjMat[k][j] < ppAdjMat[i][j])
					{
					 ppAdjMat[i][j] = ppAdjMat[i][k] + ppAdjMat[k][j];
					 ppNext[i][j] = k;
					}
				}
			}
		}
	}
}

void asp_par(int id, int **ppAdjMat, int **ppNext, int m, int n, int nProcs)
{
	int i,j,k, iProc;
	int jump = n / nProcs;

	int lb =  (id * jump) - jump; // will be 'm' except for the last processor
	int ub = lb + m - 1;
	int source, tag;

	int *RowK = (int*) malloc(sizeof(int) * n);
	if (NULL == RowK)
	{
		perror("Out of memory\n");
		return;
	}

	MPI_Status status;
	MPI_Request request;

	int count;

	for (k = 0; k < n; ++k)
	{
		if (k >= lb && k <= ub)
		{
			//
			// Processor: I have this row already.
			// Broadcast to all others. 
			// Use GLOBAL row number as TAG
			//
			StartTime = MPI_Wtime();
			memcpy(RowK, &ppAdjMat[k-lb][0], n * sizeof(int));
			for (iProc = 1; iProc <= nProcs; ++ iProc)
			{
				if (id != iProc)
				{
					//int tag = i + lb;//This is GLOBAL row number I am sending
					int tag = k;//This is GLOBAL row number I am sending
					MPI_Isend(&ppAdjMat[k-lb][0], n, MPI_INT, iProc, tag, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
				}
			}
			EndTime = MPI_Wtime();
			CommunicationTime += ((EndTime - StartTime) * 1000);
		}
		else
		{
			//
			// I know the source who will have it
			// This is very important for ordering of messages
			//
			tag = k;
			source = MIN( ((k / jump) + 1), nProcs);

			#if DEBUG_LOG
			WriteToLogf(LOG_FILE, "Process(%d) doing blocking call on row(%d.",id,k);
			#endif

			StartTime = MPI_Wtime(); 
			MPI_Recv((void*)&RowK[0], n, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_INT, &count);
			if (count != n)
			{
				#if DEBUG_LOG
				WriteToLogf(LOG_FILE,"ERROR:---count = %d, n = %d", count, n);
				#endif
			}
			EndTime = MPI_Wtime();
			CommunicationTime += ((EndTime - StartTime) * 1000);
		}//else (trying to fetch row from other processor)

		//=========================================================================
		//+
		//  Calculation of minimum
		StartTime = MPI_Wtime();
		for (i = 0; i < m; ++i)// from LB to UB
		{
			if ((i + lb) != k) // As i ranges from 0-m (local to this proc)
				// and k ranges from 0 to n( n is global)
			{
				for (j = 0; j < n; ++j)
				{
					if ((ppAdjMat[i][k] + RowK[j]) < ppAdjMat[i][j])
					{
						ppAdjMat[i][j] = ppAdjMat[i][k] + RowK[j];
						ppNext[i][j] = k;
					}
				}
			}
		}
		EndTime = MPI_Wtime();
		ComputationTime += ((EndTime - StartTime) * 1000);
		//-
		//  Calculation of minimum
		//=========================================================================

	}//k

	free (RowK);

	#if DEBUG_LOG
	WriteToLogf(LOG_FILE, "asp_par(): Process (%d): Exit", id);
	#endif
}

void
CollectRows(int **ppAdjMat, int N, int nProcs)
{
	int i;
	MPI_Status status;
	//
	// # of rows (N )  = 20
	// # of processors = 3
	//
	// Load = 20 / 3 = 6
	// Last node will carry "extra load" of 2 (20 % 3) i.e. 6 + 2 = 8
	//
	//
	// # of rows (N)   = 11
	// # of processors = 4
	//
	// Load = 11 / 4 = 2, should be 3 ( 11 % 4)
	// Last node should carry 2 ( 11 / 4)
	//
	// This wont work for N = 100, nProcs = 32
	// 100 / 32 = 3
	// 100 % 32 = 4
	// In this case we can't give 4 rows to 31 processes (will exceed 100)
	// So generalize:
	// Give load of N / nProcs to all except the last processor
	// which will handle N / nProcs + N%nProcs rows
	//
	int RowsPerProcess = N / nProcs;
	int NModuloProcs = N % nProcs;
	int LoadPerProcessor;
	int LoadForLastProcessor;

	LoadPerProcessor = RowsPerProcess;
	LoadForLastProcessor = RowsPerProcess + NModuloProcs;

	int nRows;
	int nIntegers;

	int *buffer = &ppAdjMat[0][0];

	for (i = 1; i <= nProcs; ++i)
	{
		// i is the rank of destination
		//
		if (nProcs == i)
		{
			nRows = LoadForLastProcessor;
		}
		else
		{
			nRows = LoadPerProcessor;
		}

		nIntegers = nRows * N; //Load = #of rows, load*N = rectangular block of matrix

		//
		// Tag is imp too
		// As each process is sending msg to rank 0 with its TAG
		// Here we should receive the msg with tag=process ID
		MPI_Recv(buffer, nIntegers, MPI_INT, i, i, MPI_COMM_WORLD, &status);

		buffer += (nIntegers);
	}
}


void
DistributeLoad(int **ppAdjMat, int N, int nProcs)
{
	int i;
	//
	// # of rows (N )  = 20
	// # of processors = 3
	//
	// Load = 20 / 3 = 6
	// Last node will carry "extra load" of 2 (20 % 3) i.e. 6 + 2 = 8
	//
	//
	// # of rows (N)   = 11
	// # of processors = 4
	//
	// Load = 11 / 4 = 2, should be 3 ( 11 % 4)
	// Last node should carry 2 ( 11 / 4)
	//
	int RowsPerProcess = N / nProcs;
	int NModuloProcs = N % nProcs;
	int LoadPerProcessor;
	int LoadForLastProcessor;

	LoadPerProcessor = RowsPerProcess;
	LoadForLastProcessor = RowsPerProcess + NModuloProcs;

	int nRows;
	int nIntegers;

	int *buffer = &ppAdjMat[0][0];

	for (i = 1; i <= nProcs; ++i)
	{
		// 
		// i is the rank of destination
		//
		if (nProcs == i)
		{
			nRows = LoadForLastProcessor;
		}
		else
		{
			nRows = LoadPerProcessor;
		}

		nIntegers = nRows * N; //Load = #of rows, load*N = rectangular block of matrix

		MPI_Send(&nRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//Send number of rows
		MPI_Send(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // Send number of columns

		MPI_Send(buffer, nIntegers, MPI_INT, i, 0, MPI_COMM_WORLD);

		buffer += (nIntegers);
	}
}


int main(int argc, char *argv[])
{
	int N,E;
	int **ppAdjMat;
	int **ppNext;
	int **ppLocalNext;
	int **ppLocalMatrix;
	char szFileName[MAX_PATH];

	int id;
	int iRet;
	int nProcs;
	MPI_Status status;
	MPI_Request request;
	double  dStartTime = 0.0;
	double dEndTime = 0.0;

	int distance = 0;
	int diameter = 0;
	int totalDistance = 0;
	int realDiameter = 0;
	BYTE PrintPath = 0;

	if (argc < 0 || argc > 3)
	{
		printf("Usage: asp-par <path of road_list>\n");
		return 0;
	}

	if (strlen(argv[1]) >= MAX_PATH)
	{
		printf("File name too long.\n");
		return 1;
	}

	iRet = access(argv[1], F_OK);
	if (-1 == iRet)
	{
		printf("<%s> : File does not exist\n", argv[1]);
		return 1;
	}

	if (argc == 3)
	{
	  if (argv[2][0] == 'p' || argv[2][0] == 'P')
	    PrintPath = 1;
	  else
	  {
	    printf("Invalid option: Try 'p'\n");
	    return 1;
	  }
	}

	iRet = MPI_Init(&argc, &argv);
	if (iRet != MPI_SUCCESS)
	{
		perror("Error initializing MPI\n");
		return 1;
	}

	//
	//  Get number of processes
	//
	iRet = MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	iRet = MPI_Comm_rank(MPI_COMM_WORLD, &id);

	// 
	// Rank 0 is master
	// Consider nProcs = # of workers
	//

	#if DEBUG_LOG
	unlink(LOG_FILE);
	#endif

	if (0 == id)
	{
		strcpy(szFileName, argv[1]);
		N = 0;
		E = 0;
		ppAdjMat = NULL;
		iRet = ParseFile(szFileName, &N, &E, &ppAdjMat, &ppNext);
		if (-1 == iRet)
		{
			printf("Parse error\n");
			MPI_Finalize();
			return -1;
		}

		//
		// In case of small number of processors, Do not waste time in communication
		//
		if (1 == nProcs)
		{
		  printf("Resorting to sequential...\n");
		  dStartTime = MPI_Wtime();
		  totalDistance = calculate_total_road_distance(ppAdjMat, N , N);
		  asp(ppAdjMat, ppNext, N);
		  realDiameter = calculate_diameter(ppAdjMat, N, N);
		  dEndTime = MPI_Wtime();

		  PrintAdjMatToFile(OUTPUT_FILE, ppAdjMat, N, N);
		  if (1 == PrintPath)
		  {
		    PrintPathsToFile(PATH_OUTPUT_FILE, ppAdjMat, ppNext, N, N);
		  }

		  MPI_Finalize();
		  free_int_2d(&ppAdjMat);

		  printf("File name:<%s>\n", szFileName);
		  printf("Number of vertices = %d\n", N);
		  printf("Total road distance = %d\n", totalDistance/2);
		  printf("Diameter = %d\n", realDiameter);
		  printf("%s %f milliseconds\n", "Total time = ", (dEndTime - dStartTime)*1000);

		  return 0;
		}

		nProcs -= 1;

		// If nProcs = 9 or more for N = 7, then resort to show error, because
		// parallel algorithm will have at least one idle processor and implementation of algorithm
		// will make it wait for infinite time
		if (nProcs > N)
		{
		  printf("Too many processors...\nPlease press ^C to terminate application\n");
		}

		dStartTime = MPI_Wtime();
		DistributeLoad(ppAdjMat, N, nProcs);
	}
	else
	{
		int M;
		int nMsgLength;
		MPI_Recv((void*)&M, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv((void*)&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		nProcs -= 1;

		//
		// Allocate memory for matrix in 1D array because,
		// MPI cannot handle 2D array
		//
		iRet = malloc_int_2d(&ppLocalMatrix, M, N);
		if (-1 == iRet)
		{
			return -1;
		}

		iRet = malloc_int_2d(&ppLocalNext, M, N);
		if (-1 == iRet)
		{
			free_int_2d(&ppLocalMatrix);
			return -1;
		}

		StartTime = MPI_Wtime();
		MPI_Recv((void*)(&ppLocalMatrix[0][0]), M * N, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status,MPI_INT, &nMsgLength);
		EndTime = MPI_Wtime();

		CommunicationTime += ((EndTime - StartTime) * 1000);

		// 
		// Calculate total road distance before calling parallel algorithm
		//
		StartTime = MPI_Wtime();
		distance = calculate_total_road_distance(ppLocalMatrix, M , N);
		EndTime = MPI_Wtime();
		ComputationTime += ((EndTime - StartTime) * 1000);

		asp_par(id,ppLocalMatrix, ppLocalNext, M, N, nProcs);

		StartTime = MPI_Wtime();
		diameter = calculate_diameter(ppLocalMatrix, M, N);
		EndTime = MPI_Wtime();
		ComputationTime += ((EndTime - StartTime) * 1000);

		//
		// Send my local result to rank 0, so that we can collect result there.
		//
		StartTime = MPI_Wtime();
		MPI_Isend((void*)(&ppLocalMatrix[0][0]), M * N, MPI_INT, 0, id, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		MPI_Isend((void*)(&ppLocalNext[0][0]), M * N, MPI_INT, 0, id, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		MPI_Reduce(&distance, &totalDistance, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&diameter, &realDiameter, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		EndTime = MPI_Wtime();

		CommunicationTime += ((EndTime - StartTime) * 1000);

		MPI_Reduce(&CommunicationTime, &globalCommunicationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&ComputationTime, &globalComputationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		//PrintAdjMat(ppLocalMatrix, M, N);
		free_int_2d(&ppLocalMatrix);
		free_int_2d(&ppLocalNext);
	}

	if (0 == id)
	{
		//
		// Blocking receive to collect results from all processors
		//
		#if DEBUG_LOG
		WriteToLogf(LOG_FILE, "main(): Process(%d) CollectRows", id);
		#endif

		CollectRows(ppAdjMat, N, nProcs);
		CollectRows(ppNext, N, nProcs);
		dEndTime = MPI_Wtime();
		PrintAdjMatToFile(OUTPUT_FILE, ppAdjMat, N, N);
		if (1 == PrintPath)
		{
		  PrintPathsToFile(PATH_OUTPUT_FILE, ppAdjMat, ppNext, N, N);
		}

		MPI_Reduce(&distance, &totalDistance, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&diameter, &realDiameter, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

		MPI_Reduce(&CommunicationTime, &globalCommunicationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&ComputationTime, &globalComputationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		printf("File name:<%s>\n", szFileName);
		printf("Number of vertices = %d\n", N);
		printf("Total road distance = %d\n", totalDistance/2);
		printf("Diameter = %d\n", realDiameter);


		// 
		// This was allocated in ParseFile()
		//
		free_int_2d(&ppAdjMat);
		free_int_2d(&ppNext);
		printf("Communication time = %f milliseconds\n", globalCommunicationTime);
		printf("Computation time = %f milliseconds\n", globalComputationTime);
		printf("Total time =  %f milliseconds\n", (dEndTime - dStartTime)*1000);
	}
	else
	{
		#if DEBUG_LOG
		WriteToLogf(LOG_FILE, "main(): Process(%d) Cleanup", id);
		#endif
	}


	MPI_Finalize();
	return 0;
}


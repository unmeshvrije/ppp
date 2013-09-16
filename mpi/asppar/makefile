all:	logfile asppar

logfile: ../common/common.c ../common/common.h
	gcc -o ../common/common.o -c ../common/common.c

asppar:	./asp-par.c
	mpicc -Wall -o ./asp-par ./asp-par.c ../common/common.o

clean:
	rm -f *.o

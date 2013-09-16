all:	logfile aspseq

logfile: ../common/common.c ../common/common.h
	gcc -o ../common/common.o -c ../common/common.c

aspseq:	./asp-seq.c
	mpicc -Wall -o ./asp-seq ./asp-seq.c ../common/common.o

clean:
	rm -f *.o

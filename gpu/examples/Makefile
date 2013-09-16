
debug		:= 0
include		../Makefile.inc


all: clean vectorAdd
 
vectorAdd: vectorAdd.cu
	$(NVCC) -o ./vectorAdd vectorAdd.cu ../src/Timer.cpp $(NVCCFLAGS) $(INCLUDES)

clean:
	rm -f ./vectorAdd


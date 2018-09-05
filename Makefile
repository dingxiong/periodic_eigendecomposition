CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ./
# eigen package folder
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -lm -std=c++0x 


SRC = ped.cc
SHARED = $(addprefix lib, $(SRC:.cc=.so))
STATIC = $(addprefix lib, $(SRC:.cc=.a))

all : $(SHARED) $(STATIC)
$(SHARED): $(SRC)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) \
	 -I$(EIGEN) $^ -o $@
$(SRC:.cc=.o) : $(SRC)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SRC:.cc=.o)
	$(AR) crs $@ $^
clean:
	rm -f *.so *.o *.a

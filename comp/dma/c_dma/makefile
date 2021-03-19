libdma.so : dma.o
	gcc -shared -Wl,-soname,libdma.so -o libdma.so dma.o

dma.o : dma.c
	gcc -c -fPIC dma.c -o dma.o

clean :
	-rm -vf libdma.so dma.o dma.pyc
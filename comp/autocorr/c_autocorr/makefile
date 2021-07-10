libautocorr.so : autocorr.o
	gcc -shared -Wl,-soname,libautocorr.so -o libautocorr.so autocorr.o

autocorr.o : autocorr.c
	gcc -c -fPIC autocorr.c -o autocorr.o

clean :
	-rm -vf libautocorr.so autocorr.o autocorr.pyc
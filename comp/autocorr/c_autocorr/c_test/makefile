libc_test.so : c_test.o
	gcc -shared -Wl,-soname,libc_test.so -o libc_test.so c_test.o

c_test.o : c_test.c
	gcc -c -fPIC c_test.c -o c_test.o

clean :
	-rm -vf libc_test.so c_test.o c_test.pyc
CFLAGS = -O2

autocorr: main.cpp
	g++ $(CFLAGS) -o autocorr main.cpp $(LDFLAGS)
.PHONY: test clean

test: autocorr
	./autocorr

clean:
	rm -f autocorr
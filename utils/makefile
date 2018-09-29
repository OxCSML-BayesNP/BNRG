all: gigrnd.so

gigrnd.so: gigrnd.c
	gcc -shared -lm -fPIC -O3 -o gigrnd.so gigrnd.c

clean:
	rm -f *.so

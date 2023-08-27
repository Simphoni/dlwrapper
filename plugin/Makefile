all: build/modelstat
CC := g++


build/modelstat: standalone/modelstat.cpp src/lazy_unpickler.cpp include/lazy_unpickler.h
	$(CC) -o $@ $^ -Iinclude -std=c++17
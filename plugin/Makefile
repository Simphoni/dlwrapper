all: build/modelstat
CC := g++


build/modelstat: standalone/modelstat.cpp src/lazy_unpickler.cpp
	$(CC) -o $@ $^ -Iinclude -std=c++17
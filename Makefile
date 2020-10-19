#FRF Examples Makefile
#Author: Bryan Poling
#Copyright (c) 2020 Sentek Systems, LLC. All rights reserved. 


all: HelloWorld

HelloWorld: main_HelloWorld.cpp ../FRF.cpp
	g++ -Wall -Wno-unused-function -Wno-strict-aliasing -std=c++11 -O3 -march=native -L/usr/local/lib -o HelloWorld main_HelloWorld.cpp ../FRF.cpp

clean:
	/bin/rm -f HelloWorld

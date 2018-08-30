CXX = clang++
CXXFLAGS = -std=c++1y -stdlib=libc++ -c -g -O0 -Wall -Wextra -pedantic
LD = clang++
LDFLAGS = -std=c++1y -stdlib=libc++ -lpng -lc++abi -lpthread -lm

all : mp1

mp1 : main.o PNG.o HSLAPixel.o lodepng.o mp1.o
	$(LD) main.o PNG.o HSLAPixel.o lodepng.o mp1.o $(LDFLAGS) -o mp1

main.o : main.cpp mp1.h cs225/PNG.h cs225/HSLAPixel.h
	$(CXX) $(CXXFLAGS) main.cpp

mp1.o : mp1.cpp mp1.h
	$(CXX) $(CXXFLAGS) mp1.cpp

lodepng.o : cs225/lodepng/lodepng.h cs225/lodepng/lodepng.cpp
	$(CXX) $(CXXFLAGS) cs225/lodepng/lodepng.cpp

PNG.o : cs225/PNG.cpp cs225/PNG.h cs225/lodepng/lodepng.h cs225/HSLAPixel.h
	$(CXX) $(CXXFLAGS) cs225/PNG.cpp

HSLAPixel.o : cs225/HSLAPixel.cpp cs225/HSLAPixel.h
	$(CXX) $(CXXFLAGS) cs225/HSLAPixel.cpp

test : unit_tests.o mp1.o PNG.o HSLAPixel.o lodepng.o
	$(LD) unit_tests.o mp1.o PNG.o HSLAPixel.o lodepng.o $(LDFLAGS) -o test

unit_tests.o : tests/unit_tests.cpp tests/catch.hpp cs225/PNG.h cs225/HSLAPixel.h
	$(CXX) $(CXXFLAGS) tests/unit_tests.cpp

clean :
	-rm -f mp1 test *.o

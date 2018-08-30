#include "Hash.h"
#include <string>

unsigned long bernstein(std::string str, int M)
{
	unsigned long b_hash = 5381;
	for(char c : str){
		b_hash *= 33;
		b_hash += c;
	}
	return b_hash % M;
}

std::string reverse(std::string str)
{
    std::string output = "";
		for(char c : str){
			output = c + output;
		}
	return output;
}

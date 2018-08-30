#include <vector>
#include <string>
#include "Hash.h"

using namespace std;

int hashFunction(string s, int M) {

  int sum = 0;
  for(char c : s){
    sum += c;
  }
  return sum%M;
 }

int countCollisions (int M, vector<string> inputs) {
	int collisions = 0;
	bool coll[M];
  for(int i = 0; i < M; i++){
    coll[i] = false;
  }
  for(string s : inputs){
    if(coll[hashFunction(s,M)] == true){
      collisions ++;
    }
    coll[hashFunction(s,M)] = true;
  }
	return collisions;
}

#include <vector>
#include "Primes.h"
#include <cmath>

std::vector<int> *genPrimes(int M) {
    std::vector<int> *v = new std::vector<int>();

    bool isPrime[M];
    for(int i = 0; i < M; i++){
      isPrime[i] = true;
    }
    int n = std::sqrt(M) + 1;
    for(int i = 2; i < n; i++)
    {
      if(isPrime[i])
      {
        int j = i * i;
        while(j < M){
          isPrime[j] = false;
          j += i;
        }
      }
    }

    for(int i = 2; i < M; i++){
      if(isPrime[i]){
        v->push_back(i);
      }
    }
    return v;
}

int numSequences(std::vector<int> *primes, int num) {
  int count = 0;
  int size = primes->size();
  for(int t = 0; t < size; t++)
  {
    int sum = 0;
    for(int i = t; i < num; i++)
    {
      sum += primes->at(i);
      if(sum == num){
        count++;
        break;
      }
      if(sum > num){
        break;
      }
    }
    if(primes->at(t) > num){
      break;
    }
  }
  return count;
}

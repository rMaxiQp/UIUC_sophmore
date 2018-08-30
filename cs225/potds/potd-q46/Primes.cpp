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
        int square = i * i;
        for(int j = square; j < M;)
        {
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

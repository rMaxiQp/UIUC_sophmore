#include "Friend.h"

int find(int x, std::vector<int>& parents) {
   return 1;
}

int findCircleNum(std::vector<std::vector<int>>& M) {
  int count = 0;
    for(size_t t = 0; t < M.size(); t++){
      count += find(count, M[t]);
    }
    return count;
}

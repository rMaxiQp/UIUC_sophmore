#include "MinHeap.h"

vector<int> lastLevel(MinHeap & heap)
{
  std::vector<int> v;
  int M = heap.elements.size();
  int logM = std::log2(M);
  int index = 1;
  int sum = 1;
  for(int i = 1; i < logM; i++){
    index = index * 2;
    sum = sum + index;
  }
  sum += 1;
  for(; sum < M; sum++)
  {
    v.push_back(heap.elements[sum]);
  }
  return v;
}

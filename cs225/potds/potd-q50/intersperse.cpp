#include "intersperse.h"
#include <algorithm>
#include <iostream>

std::vector<int> zigZag(std::vector<int> v1, std::vector<int> v2)
{
  std::sort(v1.begin(),v1.end());
  std::sort(v2.begin(),v2.end());
  std::vector<int> v;
  size_t a = v1.size();
  for(size_t t = 0; t < a; t++)
  {
    v.push_back(v1[a-t-1]);
    v.push_back(v2[a-t-1]);
  }
  return v;
}

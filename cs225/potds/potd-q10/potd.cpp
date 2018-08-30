#include "potd.h"
#include <cmath>

int * potd::raise(int* arr){
  int i = 0;
  int* temp = new int(*arr);
  while(arr[i] != -1){
    temp[i] = arr[i];
    i++;
  }
  temp[i] = -1;
  if(temp[0] == -1) return temp;
  i = 0;
  while(temp[i+1] != -1){
    temp[i] = std::pow(temp[i], temp[i+1]);
    i++;
  }
  return temp;
}

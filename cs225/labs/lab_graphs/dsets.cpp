#include "dsets.h"

void DisjointSets::addelements (int num) {
  for(int i = 0; i < num; i++)
  {
    sets.push_back(-1);
  }
}

int DisjointSets::find (int elem) {
  if(sets[elem] < 0) {
    return elem;
  }
  return sets[elem] = find(sets[elem]);
}

void DisjointSets::setunion (int a, int b) {
  int a_size = find(a);
  int b_size = find(b);
  if(a_size == b_size) return;
  int sum = sets[b_size] + sets[a_size];
  if(sets[a_size] <= sets[b_size]) {
    sets[b_size] = a_size;
    sets[a_size] = sum;
  }
  else {
    sets[a_size] = b_size;
    sets[b_size] = sum;
  }
}

int DisjointSets::size (int elem){
  if(sets[elem] < 0) return -sets[elem];
  return size(sets[elem]);
}

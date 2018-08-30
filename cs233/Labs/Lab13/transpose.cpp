#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "transpose.h"

// will be useful
// remember that you shouldn't go over SIZE
using std::min;

// modify this function to add tiling
void
transpose_tiled(int **src, int **dest) {
  for (int i = 0; i < SIZE;) {
    for (int j = 0; j < SIZE;) {
      for(int x = i; x < min(SIZE, i + 32); x++){
        for(int y = j; y < min(SIZE, j + 32); y++){
          dest[x][y] = src[y][x];
        }
      }
      j = j + 32;
    }
    i = i + 32;
  }
}

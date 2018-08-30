#include "utils.h"
#include <iostream>

uint32_t extract_tag(uint32_t address, const CacheConfig& cache_config) {
  uint32_t tag = cache_config.get_num_tag_bits();
  uint32_t off = 1;
  uint32_t temp = 1;
  for(uint32_t t = 1; t < tag; t++)
  {
    temp = off;
    off <<= 1;
    off |= temp;
  }
  return (address >> (32 - tag)) & off;
}

uint32_t extract_index(uint32_t address, const CacheConfig& cache_config) {
  uint32_t index = cache_config.get_num_index_bits();
  if(index == 0) return 0;
  uint32_t off = 1;
  uint32_t temp = 1;
  for(uint32_t t = 1; t < index; t++)
  {
    temp = off;
    off <<= 1;
    off |= temp;
  }
  return (address >>  cache_config.get_num_block_offset_bits()) & off;
}

uint32_t extract_block_offset(uint32_t address, const CacheConfig& cache_config) {
  uint32_t block = cache_config.get_num_block_offset_bits();
  if(block == 0) return 0;
  uint32_t off = 1;
  uint32_t temp = 1;
  for(uint32_t t = 1; t < block; t++)
  {
    temp = off;
    off <<= 1;
    off |= temp;
  }
  return address & off;
}

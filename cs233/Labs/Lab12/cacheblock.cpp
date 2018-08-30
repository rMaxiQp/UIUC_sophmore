#include "cacheblock.h"

uint32_t Cache::Block::get_address() const {
  uint32_t idx_shift = _cache_config.get_num_index_bits();
  uint32_t block_shift = _cache_config.get_num_block_offset_bits();
  return (_tag << (idx_shift + block_shift)) | (_index << block_shift);
}

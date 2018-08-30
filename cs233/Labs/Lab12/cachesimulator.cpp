#include "cachesimulator.h"

Cache::Block* CacheSimulator::find_block(uint32_t address) const {
  /**
   * TODO
   *
   * 1. Use `_cache->get_blocks_in_set` to get all the blocks that could
   *    possibly have `address` cached.
   * 2. Loop through all these blocks to see if any one of them actually has
   *    `address` cached (i.e. the block is valid and the tags match).
   * 3. If you find the block, increment `_hits` and return a pointer to the
   *    block. Otherwise, return NULL.
   */
   vector<Cache::Block*> potential = _cache->get_blocks_in_set(extract_index(address,_cache->get_config()));
   uint32_t tag = extract_tag(address, _cache->get_config());
   for(size_t t = 0; t < potential.size(); t++)
   {
     if(potential[t]->is_valid() && potential[t]->get_tag() == tag)
     {
       _hits++;
       return potential[t];
     }
   }
  return NULL;
}

Cache::Block* CacheSimulator::bring_block_into_cache(uint32_t address) const {
  /**
   * TODO
   *
   * 1. Use `_cache->get_blocks_in_set` to get all the blocks that could
   *    cache `address`.
   * 2. Loop through all these blocks to find an invalid `block`. If found,
   *    skip to step 4.
   * 3. Loop through all these blocks to find the least recently used `block`.
   *    If the block is dirty, write it back to memory.
   * 4. Update the `block`'s tag. Read data into it from memory. Mark it as
   *    valid. Mark it as clean. Return a pointer to the `block`.
   */
   vector<Cache::Block*> potential = _cache->get_blocks_in_set(extract_index(address, _cache->get_config()));
  uint32_t lru = 0;
  uint32_t lru_time = INT32_MAX;
  uint32_t tag = extract_tag(address, _cache->get_config());
  for(size_t t = 0; t < potential.size(); t++)
  {
    if(!potential[t]->is_valid())
    {
      potential[t]->set_tag(tag);
      potential[t]->read_data_from_memory(_memory);
      potential[t]->mark_as_valid();
      potential[t]->mark_as_clean();
      return potential[t];
    }
    if(lru_time > potential[t]->get_last_used_time())
    {
      lru_time = potential[t]->get_last_used_time();
      lru = t;
    }
  }

  if(potential[lru]->is_dirty())
  {
    potential[lru]->write_data_to_memory(_memory);
  }
  potential[lru]->set_tag(tag);
  potential[lru]->read_data_from_memory(_memory);
  potential[lru]->mark_as_valid();
  potential[lru]->mark_as_clean();
  return potential[lru];
}

uint32_t CacheSimulator::read_access(uint32_t address) const {
  /**
   * TODO
   *
   * 1. Use `find_block` to find the `block` caching `address`.
   * 2. If not found, use `bring_block_into_cache` cache `address` in `block`.
   * 3. Update the `last_used_time` for the `block`.
   * 4. Use `read_word_at_offset` to return the data at `address`.
   */
  Cache::Block* temp = find_block(address);
  if(temp == NULL)
  {
    temp = bring_block_into_cache(address);
  }
  _use_clock++;
  temp->set_last_used_time(_use_clock.get_count());
  return temp->read_word_at_offset(extract_block_offset(address, _cache->get_config()));
}

void CacheSimulator::write_access(uint32_t address, uint32_t word) const {
  /**
   * TODO
   *
   * 1. Use `find_block` to find the `block` caching `address`.
   * 2. If not found
   *    a. If the policy is write allocate, use `bring_block_into_cache`.
   *    a. Otherwise, directly write the `word` to `address` in the memory
   *       using `_memory->write_word` and return.
   * 3. Update the `last_used_time` for the `block`.
   * 4. Use `write_word_at_offset` to to write `word` to `address`.
   * 5. a. If the policy is write back, mark `block` as dirty.
   *    b. Otherwise, write `word` to `address` in memory.
   */
   Cache::Block* temp = find_block(address);
   if(temp == NULL)
   {
     if(_policy.is_write_allocate())
     {
       temp = bring_block_into_cache(address);
     }
     else
     {
       _memory->write_word(address, word);
       return;
     }
   }
   _use_clock++;
   temp->set_last_used_time(_use_clock.get_count());
   temp->write_word_at_offset(word, extract_block_offset(address, _cache->get_config()));
   if(_policy.is_write_back())
   {
     temp->mark_as_dirty();
   }
   else
   {
     temp->write_data_to_memory(_memory);
   }
}

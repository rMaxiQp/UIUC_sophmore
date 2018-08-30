#include "simplecache.h"
using namespace std;

int SimpleCache::find(int index, int tag, int block_offset) {
  if(index < _cache.size())
  {
    vector<SimpleCacheBlock> &temp = _cache[index];
    for(size_t t = 0; t < temp.size(); t++)
    {
      if(temp[t].valid() && temp[t].tag() == tag)
      {
        return (int)temp[t].get_byte(block_offset);
      }
    }
  }
  return 0xdeadbeef;
}

void SimpleCache::insert(int index, int tag, char data[]) {
  if(index < _cache.size())
  {
    for(size_t t = 0; t < _cache[index].size(); t++)
    {
      if(!_cache[index].at(t).valid())
      {
        _cache[index].at(t).replace(tag, data);
        return;
      }
    }
    _cache[index].at(0).replace(tag, data);
  }
}

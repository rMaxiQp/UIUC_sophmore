
/**
 * @file heap.cpp
 * Implementation of a heap class.
 */

#include <algorithm>

template <class T, class Compare>
size_t heap<T, Compare>::root() const
{
  return 1;
}

template <class T, class Compare>
size_t heap<T, Compare>::leftChild(size_t currentIdx) const
{
  return 2 * currentIdx;
}

template <class T, class Compare>
size_t heap<T, Compare>::rightChild(size_t currentIdx) const
{
  return 2 * currentIdx + 1;
}

template <class T, class Compare>
size_t heap<T, Compare>::parent(size_t currentIdx) const
{
  return currentIdx / 2 ;
}

template <class T, class Compare>
bool heap<T, Compare>::hasAChild(size_t currentIdx) const
{
  return 2 * currentIdx < _elems.size();
}

template <class T, class Compare>
size_t heap<T, Compare>::maxPriorityChild(size_t currentIdx) const
{
  return higherPriority(_elems[2 * currentIdx], _elems[2 * currentIdx + 1]) ?
     2 * currentIdx : 2 * currentIdx + 1;
}

template <class T, class Compare>
void heap<T, Compare>::heapifyDown(size_t currentIdx)
{
  if(!hasAChild(currentIdx)) return;

  size_t length = _elems.size();
  size_t left = 2 * currentIdx;
  size_t right = 2 * currentIdx + 1;
  size_t minIndex = currentIdx;
  size_t hold = maxPriorityChild(minIndex);

  if(hold < length && _elems[hold] < _elems[minIndex]){
    minIndex = hold;
  }

  if (minIndex != currentIdx)
  {
    std::swap(_elems[minIndex], _elems[currentIdx]);
    heapifyDown(minIndex);
  }
}

template <class T, class Compare>
void heap<T, Compare>::heapifyUp(size_t currentIdx)
{
    if (currentIdx == root())
        return;
    size_t parentIdx = parent(currentIdx);
    if (higherPriority(_elems[currentIdx], _elems[parentIdx])) {
        std::swap(_elems[currentIdx], _elems[parentIdx]);
        heapifyUp(parentIdx);
    }
}

template <class T, class Compare>
heap<T, Compare>::heap()
{
  _elems.push_back(T());
    /// @todo Depending on your implementation, this function may or may
    ///   not need modifying
}

template <class T, class Compare>
heap<T, Compare>::heap(const std::vector<T>& elems)
{
  _elems.push_back(T());
  for(auto i : elems)
  {
    _elems.push_back(i);
  }
  for(size_t t = parent(_elems.size()); t > 0; t--)
  {
    heapifyDown(t);
  }
}

template <class T, class Compare>
T heap<T, Compare>::pop()
{
  size_t length = _elems.size();
  T p = T();
  if(length > 1)
  {
    p = _elems[1];
    _elems[1] = _elems[length-1];
    _elems.pop_back();
    heapifyDown(1);
  }
  return p;
}

template <class T, class Compare>
T heap<T, Compare>::peek() const
{
  return _elems[1];
}

template <class T, class Compare>
void heap<T, Compare>::push(const T& elem)
{
  _elems.push_back(elem);
  heapifyUp(_elems.size()-1);
}

template <class T, class Compare>
bool heap<T, Compare>::empty() const
{
  return _elems.size() == 1;
}

template <class T, class Compare>
void heap<T, Compare>::getElems(std::vector<T> & heaped) const
{
    for (size_t i = root(); i < _elems.size(); i++) {
        heaped.push_back(_elems[i]);
    }
}

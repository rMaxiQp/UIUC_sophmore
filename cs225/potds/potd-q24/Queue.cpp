#include "Queue.h"
#include <iostream>
using namespace std;
Queue::Queue() { }

// `int size()` - returns the number of elements in the stack (0 if empty)
int Queue::size() const {
  return size_;
}

// `bool isEmpty()` - returns if the list has no elements, else false
bool Queue::isEmpty() const {
  return size_ == 0;
}

// `void enqueue(int val)` - enqueue an item to the queue in O(1) time
void Queue::enqueue(int value) {
  Queue* current = new Queue(value);
  if(size_ == 0){
    tail_ = current;
    head_ = current;
  }else{
    tail_->next_ = current;
    tail_ = current;
  }
  size_++;
}

// `int dequeue()` - removes an item off the queue and returns the value in O(1) time
int Queue::dequeue() {
  if(size_ == 0) return 0;
  int temp = head_->data_;
  head_ = head_->next_;
  size_--;
  return temp;
}

Queue::Queue(int value){
  data_ = value;
  next_ = NULL;
}

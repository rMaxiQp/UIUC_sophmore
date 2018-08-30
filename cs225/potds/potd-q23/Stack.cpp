#include "Stack.h"
#include <iostream>
using namespace std;
// `int size()` - returns the number of elements in the stack (0 if empty)
int Stack::size() const {
  return size_;
}

// `bool isEmpty()` - returns if the list has no elements, else false
bool Stack::isEmpty() const {
  return size_ == 0;
}

// `void push(int val)` - pushes an item to the stack in O(1) time
void Stack::push(int value) {
  if(size_ == 0){
    head_ = new Stack(value);
  }else{
    current = new Stack(value);
    current->next_ = head_;
    head_ = current;
  }
  size_++;
}

// `int pop()` - removes an item off the stack and returns the value in O(1) time
int Stack::pop() {
  if(size_ == 0) return 0;
  int value = head_->data_;
  head_ = head_->next_;
  size_ --;
  return value;
}

Stack::Stack(int value){
  this->data_ = value;
  this->next_ = NULL;
}

Stack::Stack(){

}

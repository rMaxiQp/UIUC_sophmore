#ifndef _STACK_H
#define _STACK_H

#include <cstddef>

class Stack {
public:
  int size() const;
  bool isEmpty() const;
  void push(int value);
  int pop();
  Stack();
  Stack(int value);
  Stack* next_;
  Stack* head_;
  Stack* current;
private:
  int size_ = 0;
  int data_;
};

#endif

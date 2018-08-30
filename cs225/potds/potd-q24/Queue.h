#ifndef _QUEUE_H
#define _QUEUE_H

#include <cstddef>

class Queue {
    public:
        int size() const;
        bool isEmpty() const;
        void enqueue(int value);
        int dequeue();
        Queue();
        Queue(int value);
    private:
      Queue* head_;
      Queue* tail_;
      Queue* next_;
      int data_;
      int size_ = 0;
};

#endif

#include "Node.h"
using namespace std;

void sortList(Node **head) {
  if(Node::getNumNodes()<1) return;

  Node* curSmall = *head;
  Node* pointer = *head;
  Node* cur = *head;
  Node* pointerPrev = NULL;
  Node* curPrev = NULL;
  Node* curSmallprev = NULL;

for(int i = 0; i < Node::getNumNodes()-1; i++)
  {
    while(cur != NULL)//find the smallest value in the list
    {
      if(curSmall->data_ > cur->data_)
      {
        curSmallprev = curPrev;
        curSmall = cur;
      }
      curPrev = cur;
      cur = cur->next_;
    }

    if(i == 0)//the first sort
    {
      *head = curSmall;
    }
    if(curSmall != pointer)//if we find smaller value than pointer
    {
      if(curSmallprev != NULL) curSmallprev->next_ = curSmall->next_;
      curSmall->next_ = pointer;
      if(pointerPrev != NULL) pointerPrev->next_ = curSmall;
      pointerPrev = curSmall;
    }else{
      pointerPrev = pointer;
      curSmallprev = pointer;
      pointer = pointer->next_;
    }
    curSmall = pointer;
    cur = pointer;
  }
}

Node::Node() {
    numNodes++;
}

Node::Node(Node &other) {
    this->data_ = other.data_;
    this->next_ = other.next_;
    numNodes++;
}

Node::~Node() {
    numNodes--;
}

int Node::numNodes = 0;

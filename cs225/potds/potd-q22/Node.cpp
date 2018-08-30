#include "Node.h"
#include <iostream>
using namespace std;

Node *listSymmetricDifference(Node *first, Node *second) {
if(first==NULL && second == NULL) return NULL;
else if(first == NULL) return second;
else if(second == NULL) return first;

int numNodes = first->getNumNodes();
int first_[numNodes];
int second_[numNodes];
int firstNum = 0;
int secondNum = 0;
Node* f = first;
Node* s = second;

while(f){
  first_[firstNum] = f->data_;
  firstNum ++;
  f = f->next_;
}
while(s){
  second_[secondNum] = s->data_;
  secondNum ++;
  s = s->next_;
}

int num = 0;
int arr[max(firstNum, secondNum)];
for(int i=0; i<firstNum; i++){
  bool same = false;
  for(int j =0; j<secondNum; j++){
    if(first_[i] == second_[j]){
      same = true;
      break;
    }
  }
  if(!same){
    arr[num++] = first_[i];
  }
}
for(int i=0; i<secondNum; i++){
  bool same = false;
  for(int j =0; j<firstNum; j++){
    if(first_[j] == second_[i]){
      same = true;
      break;
    }
  }
  if(!same){
    arr[num++] = second_[i];
  }
}

if(num == 0) return NULL;

Node* result = NULL;
Node* point = NULL;

for(int i =0; i<num; i++){
  bool same = false;
  for(int j = i; j<num; j++){
    if(arr[i] == arr[j] && i != j){
      same = true;
      break;
    }
  }
  if(!same){
    if(result == NULL){
      result = new Node();
      result->data_ = arr[i];
      point = result;
    }
    else
    {
      point->next_ = new Node();
      point = point->next_;
      point->data_ = arr[i];
    }
  }
}
  point->next_ = NULL;
  return result;
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

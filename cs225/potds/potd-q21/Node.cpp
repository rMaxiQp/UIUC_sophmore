#include "Node.h"
#include <iostream>
using namespace std;

Node *listIntersection(Node *first, Node *second) {
  if(first == NULL || second == NULL) return NULL;
  Node* newOne = NULL;
  Node* start = NULL;
  Node* temp = NULL;
  Node* f = first;
  Node* s = second;

  while(f){
    while(s){
      if(f->data_ == s->data_){
        if(start==NULL) {
          newOne = new Node();
          newOne->data_ = f->data_;
          start = newOne;
          cout<<start->data_<<endl;
        }
        else{
         for(temp = newOne; temp; temp = temp->next_){
           if(temp->data_ == f->data_){
             break;
           }
         }
         if(temp == NULL){
          start->next_ = new Node();
          start = start->next_;
          start->data_ = f->data_;
          cout<<start->data_<<endl;
         }
       }
    }
        s = s->next_;

    }
    f = f->next_;
    if(f==NULL && s==NULL) break;
    s = second;
}
temp = newOne;
  while(temp){
    std::cout << temp->data_ << '\n';
    temp = temp->next_;
  }
  if(start)start->next_ = NULL;
  return newOne;
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

#include <vector>
#include "BTreeNode.h"


BTreeNode* find(BTreeNode* root, int key) {
  BTreeNode* temp = NULL;
  for(unsigned long t = 0; t < root->elements_.size(); t++){
    if(root->elements_[t] == key) return root;
  }
  for(unsigned long t = 0; t < root->children_.size(); t++){
    temp = find(root->children_[t], key);
    if(temp != NULL) return temp;
  }
  return temp;
}

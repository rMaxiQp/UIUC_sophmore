#include "TreeNode.h"

#include <cstddef>
#include <iostream>
using namespace std;

TreeNode::TreeNode() : left_(NULL), right_(NULL) { }

int TreeNode::getHeight() {
  if(this->left_ == NULL && this->right_ == NULL) return 0;
  int left = 0;
  int right = 0;
  if(this->left_) left = this->left_->getHeight();
  if(this->right_) right = this->right_->getHeight();
  return 1 + max(left, right);
}

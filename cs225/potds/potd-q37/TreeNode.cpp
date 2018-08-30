#include "TreeNode.h"
#include <algorithm>
#include <iostream>
using namespace std;

TreeNode* findLastUnbalanced(TreeNode* root) {
  updated(root);
  TreeNode* temp = find(root->left_);
  TreeNode* hold = find(root->right_);
  if(getHeigth(temp) > getHeigth(hold)) return temp;
  if(getHeigth(temp) < getHeigth(hold))return hold;
  if(getHeigth(root->left_) - getHeigth(root->right_) > 1 || getHeigth(root->left_) - getHeigth(root->right_) < -1) return root;
  return NULL;
}

int getHeigth(TreeNode* subroot){
  if(!subroot) return -1;
  return subroot->balance_;
}

TreeNode* find(TreeNode* subroot){
  if(!subroot) return NULL;
  find(subroot->left_);
  find(subroot->right_);
  TreeNode* temp = subroot->left_;
  TreeNode* hold = subroot->right_;
  if(getHeigth(temp) - getHeigth(hold) > 1 || getHeigth(temp) - getHeigth(hold) < -1) {
    return subroot;
  }

  return NULL;
}

void updated(TreeNode* subroot){
  cout<<"SUBROOT:: "<< subroot->val_<<" :: "<< subroot<<endl;
  if(subroot->left_  == NULL &&  subroot->right_ == NULL)
  {
    subroot->balance_ = 0;
    return;
  }

 if(subroot->right_ != NULL) updated(subroot->right_);
 if(subroot->left_ != NULL) updated(subroot->left_);

 if(subroot->right_ != NULL && subroot->left_ != NULL) subroot->balance_ = 1 + max(subroot->left_->balance_, subroot->right_->balance_);
 else if(subroot->right_ != NULL) subroot->balance_ = 1 + subroot->right_->balance_;
 else subroot->balance_ = 1 + subroot->left_->balance_;
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

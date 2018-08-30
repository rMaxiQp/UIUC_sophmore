#include "TreeNode.h"
#include <algorithm>
#include <iostream>

using namespace std;

void swap(TreeNode*& first, TreeNode*& second)
{
    int tempkey = first->val_;
    first->val_ = second->val_;
    second->val_ = tempkey;
}


void rightRotate(TreeNode* root) {
  TreeNode* temp = root->left_;
  cout<<"PREV root->parent_: "<< root->parent_<< " "<<endl;
  cout<<"PREV temp->parent_: "<< temp->parent_<<" "<<endl;
  root->left_ = temp->right_;
  if(temp->right_ )temp->right_->parent_ = root;
  temp->parent_ = root->parent_;
  root->parent_ = temp;
  temp->right_ = root;
  //root = temp;
  cout<<"root->parent_: "<< root->parent_<< " " <<endl;
  cout<<"temp->parent_: "<< temp->parent_<<" "<<endl;
}


void leftRotate(TreeNode* root) {
  TreeNode* temp = root->right_;
  cout<<"PREV root->parent_: "<< root->parent_<< " "<<endl;
  cout<<"PREV temp->parent_: "<< temp->parent_<<" "<<endl;
  root->right_ = temp->left_;
  if(temp->left_) temp->left_->parent_ =root;
  temp->parent_ = root->parent_;
  root->parent_ = temp;
  temp->left_ = root;
  //root = temp;
  cout<<"root->parent_: "<< root->parent_<< " "<<endl;
  cout<<"temp->parent_: "<< temp->parent_<<" "<<endl;
}



void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

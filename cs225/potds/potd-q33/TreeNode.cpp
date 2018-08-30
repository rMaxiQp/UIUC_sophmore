#include "TreeNode.h"
#include <iostream>

TreeNode * deleteNode(TreeNode* root, int key) {
  inorderPrint(root);
  TreeNode* subroot = root;
  TreeNode* smallest = NULL;
  TreeNode* prevNode = NULL;
  TreeNode* smallestPrev = NULL;
  while(subroot != NULL){
    if(subroot->val_ > key)
    {
      prevNode = subroot;
      subroot = subroot->left_;
    }
    else if (subroot->val_ < key)
    {
      prevNode = subroot;
      smallestPrev = prevNode;
      //std::cout<<"SP: "<<smallestPrev->val_<<'\n';
      smallest = subroot;
      subroot = subroot->right_;
    }
    else if (subroot->val_ == key)
    {
      if(subroot->right_)
      {
        //std::cout<<"SP: "<<smallestPrev->val_<<'\n';
        smallestPrev = smallest;
        smallest = subroot->right_;
        while(smallest->left_){
          if(smallest->left_ != NULL) {
            smallestPrev = smallest;
            smallest = smallest->left_;
          }
        }
      }
      break;
    }
  }
  if(subroot == NULL) return root;
  //if(subroot == root) root = smallest;
  if(subroot->left_ && subroot->right_)//delete 2
  {
    int temp = smallest->val_;
    smallest->val_ = subroot->val_;
    subroot->val_ = temp;
    //std::cout << "/* message */" << '\n';
    if(subroot->right_ == smallest){
      subroot->right_ = smallest->right_;
    }
    else{
      smallestPrev->left_ = NULL;
    }
  }
  else if(subroot->left_ || subroot->right_)
  {
    if(prevNode->left_ == subroot){
      if(subroot->left_)
      {
        prevNode->left_ = subroot->left_;
      }
      else
      {
        prevNode->left_ = subroot->right_;
      }
    }
    else
    {
      if(subroot->left_)
      {
        prevNode->right_ = subroot->left_;
      }
    else
      {
        prevNode->right_ = subroot->right_;
      }
    }
  }
  else{
    if(prevNode->left_ == subroot) prevNode->left_ = NULL;
    else prevNode->right_ = NULL;
  }
  //inorderPrint(root);
  return root;
}

void inorderPrint(TreeNode* node)
{
    if (!node)  return;
    inorderPrint(node->left_);
    std::cout << node->val_ << " ";
    inorderPrint(node->right_);
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

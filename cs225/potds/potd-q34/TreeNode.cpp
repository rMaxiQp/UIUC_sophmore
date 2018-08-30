#include "TreeNode.h"

int getHeightBalance(TreeNode* root) {
  if(!root) return 0;
  return getHeigth(root->left_) - getHeigth(root->right_);
}

int getHeigth(TreeNode* subroot){
  if(!subroot) return 0;
  return 1 + (getHeigth(subroot->left_) <  getHeigth(subroot->right_) ? getHeigth(subroot->right_) : getHeigth(subroot->left_));
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

#include "TreeNode.h"

int getHeigth(TreeNode* subroot){
  if(!subroot) return 0;
  return 1 + (getHeigth(subroot->left_) <  getHeigth(subroot->right_) ? getHeigth(subroot->right_) : getHeigth(subroot->left_));
}

bool isHeightBalanced(TreeNode* root) {
  if(!root) return true;
  int temp = getHeigth(root->left_) - getHeigth(root->right_);
  if(temp > 1 || temp < -1) return false;
  return true;
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

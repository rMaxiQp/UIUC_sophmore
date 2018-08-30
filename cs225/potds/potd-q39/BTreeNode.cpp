#include <vector>
#include "BTreeNode.h"


std::vector<int> traverse(BTreeNode* root) {
    std::vector<int> v;
    if(!root->is_leaf_){
      for(int t = 0; t <= root->elements_.size(); t++){
        std::vector<int> k = traverse(root->children_[t]);
      for(int a = 0; a < k.size(); a++) {
          v.push_back(k[a]);
        }
        if(t < root->elements_.size())v.push_back(root->elements_[t]);
      }
    }
    else v = root->elements_;
    return v;
}

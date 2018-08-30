/**
 * @file binarytree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */

/**
 * @return The height of the binary tree. Recall that the height of a binary
 *  tree is just the length of the longest path from the root to a leaf, and
 *  that the height of an empty tree is -1.
 */
template <typename T>
int BinaryTree<T>::height() const
{
    // Call recursive helper function on root
    return height(root);
}

/**
 * Private helper function for the public height function.
 * @param subRoot
 * @return The height of the subtree
 */
template <typename T>
int BinaryTree<T>::height(const Node* subRoot) const
{
    // Base case
    if (subRoot == NULL)
        return -1;

    // Recursive definition
    return 1 + max(height(subRoot->left), height(subRoot->right));
}

/**
 * Prints out the values of the nodes of a binary tree in order.
 * That is, everything to the left of a node will be printed out before that
 * node itself, and everything to the right of a node will be printed out after
 * that node.
 */
template <typename T>
void BinaryTree<T>::printLeftToRight() const
{
    // Call recursive helper function on the root
    printLeftToRight(root);

    // Finish the line
    cout << endl;
}

/**
 * Private helper function for the public printLeftToRight function.
 * @param subRoot
 */
template <typename T>
void BinaryTree<T>::printLeftToRight(const Node* subRoot) const
{
    // Base case - null node
    if (subRoot == NULL)
        return;

    // Print left subtree
    printLeftToRight(subRoot->left);

    // Print this node
    cout << subRoot->elem << ' ';

    // Print right subtree
    printLeftToRight(subRoot->right);
}

/**
 * Flips the tree over a vertical axis, modifying the tree itself
 *  (not creating a flipped copy).
 */
template <typename T>
void BinaryTree<T>::mirror()
{
    mirror(root);
}

template <typename T>
void BinaryTree<T>::mirror(Node* root){
  if(root == NULL) return;
  Node* temp = root;
  temp = root->right;
  root->right = root->left;
  root->left = temp;
  mirror(root->left);
  mirror(root->right);
}

/**
 * @return True if an in-order traversal of the tree would produce a
 *  nondecreasing list output values, and false otherwise. This is also the
 *  criterion for a binary tree to be a binary search tree.
 */
template <typename T>
bool BinaryTree<T>::isOrdered() const
{
  return isOrdered(root->left, root->elem, 1) && isOrdered(root->right, root->elem, 0);
}

template <typename T>
bool BinaryTree<T>::isOrdered(Node* subRoot, T hold, bool isLeft) const
{
  if(subRoot == NULL) return true;
  if(isLeft){
    if(subRoot->elem > hold) return false;
  }else{
    if(subRoot->elem < hold) return false;
  }
  return isOrdered(subRoot->left, subRoot->elem, 1) && isOrdered(subRoot->right, subRoot->elem, 0);

}

/**
 * creates vectors of all the possible paths from the root of the tree to any leaf
 * node and adds it to another vector.
 * Path is, all sequences starting at the root node and continuing
 * downwards, ending at a leaf node. Paths ending in a left node should be
 * added before paths ending in a node further to the right.
 * @param paths vector of vectors that contains path of nodes
 */
template <typename T>
void BinaryTree<T>::printPaths(vector<vector<T> > &paths) const
{
  vector<T> temp;
  printPaths(root, paths, temp);
}

template <typename T>
void BinaryTree<T>::printPaths(Node* subRoot, vector<vector<T>> &track, vector<T> temp) const
{
  temp.push_back(subRoot->elem);
  if(subRoot->left == NULL && subRoot->right == NULL){
     track.push_back(temp);
     return;
   }
  if(subRoot->left) printPaths(subRoot->left, track, temp);
  if(subRoot->right) printPaths(subRoot->right, track,temp);

}

/**
 * Each node in a tree has a distance from the root node - the depth of that
 * node, or the number of edges along the path from that node to the root. This
 * function returns the sum of the distances of all nodes to the root node (the
 * sum of the depths of all the nodes). Your solution should take O(n) time,
 * where n is the number of nodes in the tree.
 * @return The sum of the distances of all nodes to the root
 */
template <typename T>
int BinaryTree<T>::sumDistances() const
{
    return sumDistances(root, 0) - 1;
}

template <typename T>
int BinaryTree<T>::sumDistances(Node* subRoot, int sum) const
{
  if(subRoot == NULL) return 0;
  if(subRoot-> left== NULL && subRoot->right == NULL) return sum;
  return 1 + sumDistances(subRoot->left, sum + 1) + sumDistances(subRoot->right, sum + 1);
}

/**
 * @file avltree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */

template <class K, class V>
V AVLTree<K, V>::find(const K& key) const
{
    return find(root, key);
}

template <class K, class V>
V AVLTree<K, V>::find(Node* subtree, const K& key) const
{
    if (subtree == NULL)
        return V();
    else if (key == subtree->key)
        return subtree->value;
    else {
        if (key < subtree->key)
            return find(subtree->left, key);
        else
            return find(subtree->right, key);
    }
}

template <class K, class V>
void AVLTree<K, V>::rotateLeft(Node*& t)
{
    functionCalls.push_back("rotateLeft"); // Stores the rotation name (don't remove this)
    Node* temp = t->right;
    swap(temp, t);
    t->right = temp->right;
    temp->right = temp->left;
    temp->left = t->left;
    t->left = temp;
    temp->height = 1 + max(heightOrNeg1(temp->left), heightOrNeg1(temp->right));
    t->height = 1 + max(heightOrNeg1(t->left), heightOrNeg1(t->right));
}

template <class K, class V>
void AVLTree<K, V>::rotateLeftRight(Node*& t)
{
    functionCalls.push_back("rotateLeftRight"); // Stores the rotation name (don't remove this)
    // Implemented for you:
    rotateLeft(t->left);
    rotateRight(t);
}

template <class K, class V>
void AVLTree<K, V>::rotateRight(Node*& t)
{
    functionCalls.push_back("rotateRight"); // Stores the rotation name (don't remove this)
    Node* temp = t->left;
    swap(temp, t);
    t->left = temp->left;
    temp->left = temp->right;
    temp->right = t->right;
    t->right = temp;
    temp->height = 1 + max(heightOrNeg1(temp->left), heightOrNeg1(temp->right));
    t->height = 1 + max(heightOrNeg1(t->left), heightOrNeg1(t->right));
}

template <class K, class V>
void AVLTree<K, V>::rotateRightLeft(Node*& t)
{
    functionCalls.push_back("rotateRightLeft"); // Stores the rotation name (don't remove this)
    rotateRight(t->right);
    rotateLeft(t);
}

template <class K, class V>
void AVLTree<K, V>::rebalance(Node*& subtree)
{
  if(!subtree) return;
  int leftHeight = heightOrNeg1(subtree->left);
  int rightHeight = heightOrNeg1(subtree->right);
  if(leftHeight - rightHeight > 1) //left is larger
  {
    if(heightOrNeg1(subtree->left->right) - heightOrNeg1(subtree->left->left) > 0) {
      rotateLeftRight(subtree);
    }
    else {
      rotateRight(subtree);
    }
  }
  else if(rightHeight - leftHeight > 1) //right is larger
  {
    if(heightOrNeg1(subtree->right->left) - heightOrNeg1(subtree->right->right) > 0){
      rotateRightLeft(subtree);
    }
    else {
      rotateLeft(subtree);
    }
  }
}

template <class K, class V>
void AVLTree<K, V>::insert(const K & key, const V & value)
{
    insert(root, key, value);
}

template <class K, class V>
void AVLTree<K, V>::insert(Node*& subtree, const K& key, const V& value)
{
  if(!subtree){
    subtree = new Node(key,value);
    subtree->left = NULL;
    subtree->right = NULL;
  }
  else if(subtree->key < key)
  {
    if (subtree->right)
    {
      insert(subtree->right, key, value);
    }
    else
    {
      subtree->right = new Node(key,value);
      subtree->right->height = 0;
    }
  }
  else if (subtree->key > key)
  {
    if (subtree->left)
    {
      insert(subtree->left, key, value);
    }
    else
    {
      subtree->left = new Node(key, value);
      subtree->left->height = 0;
    }
  }
  subtree->height = 1 + max(heightOrNeg1(subtree->left), heightOrNeg1(subtree->right));

  rebalance(subtree);
}

template <class K, class V>
void AVLTree<K, V>::remove(const K& key)
{
    remove(root, key);

}

template <class K, class V>
void AVLTree<K, V>::remove(Node*& subtree, const K& key)
{
    if (subtree == NULL)
        return;

    if (key < subtree->key) {
        remove(subtree->left, key);
    } else if (key > subtree->key) {
        remove(subtree->right, key);
    } else {
        if (subtree->left == NULL && subtree->right == NULL) {
            /* no-child remove */
            delete subtree;
            subtree = NULL;
            return;
        } else if (subtree->left != NULL && subtree->right != NULL) {
            /* two-child remove */
            Node* temp = subtree->left;
            Node* prev = subtree;
            while(temp->right){
              prev = temp;
              temp = temp->right;
            }
            swap(temp, subtree);
            if(prev != subtree) prev->right = temp->left;
            else subtree->left = temp->left;
            delete temp;
            temp = NULL;
        } else {
            /* one-child remove */
            if(subtree->right)
            {
              swap(subtree->right, subtree);
              Node* temp = subtree->right;
              subtree->left = temp->left;
              subtree->right = temp->right;
              delete temp;
              temp = NULL;
            }
            else if(subtree->left)
            {
              swap(subtree->left, subtree);
              Node* temp = subtree->left;
              subtree->left = temp->left;
              subtree->right = temp->right;
              delete temp;
              temp = NULL;
            }
        }
    }

    subtree->height = 1 + max(heightOrNeg1(subtree->left), heightOrNeg1(subtree->right));
    rebalance(subtree);
}

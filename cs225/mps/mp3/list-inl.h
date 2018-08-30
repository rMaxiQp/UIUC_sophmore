#include <iostream>
/**
 * @file list.cpp
 * Doubly Linked List (MP 3).
 */

/**
 * Destroys the current List. This function should ensure that
 * memory does not leak on destruction of a list.
 */
template <class T>
List<T>::~List() {
  clear();
}

/**
 * Destroys all dynamically allocated memory associated with the current
 * List class.
 */
template <class T>
void List<T>::clear() {
  while(head_){
    ListNode *temp = head_->next;
    head_->prev = NULL;
    delete head_;
    head_ = temp;
  }
}

/**
 * Inserts a new node at the front of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <class T>
void List<T>::insertFront(T const& ndata) {

  ListNode* newNode = new ListNode(ndata);
  if(head_ == NULL){
    head_ = newNode;
    tail_ = newNode;
  }
  else{
    newNode->next = head_;
    head_->prev = newNode;
    head_  = newNode;
  }
  length_++;
}

/**
 * Inserts a new node at the back of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <class T>
void List<T>::insertBack(const T& ndata) {
  ListNode* newNode = new ListNode(ndata);
  if(tail_ == NULL){
    head_ = newNode;
    tail_ = newNode;
  }
  else{
    newNode->prev = tail_;
    tail_->next = newNode;
    tail_ = newNode;
  }
  length_++;
}

/**
 * Reverses the current List.
 */
template <class T>
void List<T>::reverse() {
  reverse(head_, tail_);
}

/**
 * Helper function to reverse a sequence of linked memory inside a List,
 * starting at startPoint and ending at endPoint. You are responsible for
 * updating startPoint and endPoint to point to the new starting and ending
 * points of the rearranged sequence of linked memory in question.
 *
 * @param startPoint A pointer reference to the first node in the sequence
 *  to be reversed.
 * @param endPoint A pointer reference to the last node in the sequence to
 *  be reversed.
 */
template <class T>
void List<T>::reverse(ListNode*& startPoint, ListNode*& endPoint) {
  if(startPoint == endPoint) return;
  if(startPoint == NULL || endPoint == NULL) return;

  ListNode* temp;
  ListNode* current = startPoint;

  ListNode* startPrev = startPoint->prev;
  ListNode* endNext = endPoint->next;

  while(current != endPoint){
    temp = current->next;
    current->next = current->prev;
    current->prev = temp;
    current = current->prev;
  }

  /*switch pointers of endpoint*/
  temp = endPoint->next;
  endPoint->next = current->prev;
  current->prev = temp;
  /*switch startPoint->prev and endPoint->next*/
  temp = endPoint->prev;
  endPoint->prev = startPoint->next;
  startPoint->next = temp;

  if(startPrev) startPrev->next = endPoint;

  if(endNext) endNext->prev = startPoint;
  temp = endPoint;
  endPoint = startPoint;
  startPoint = temp;

  temp = NULL;
  current = NULL;
}

/**
 * Reverses blocks of size n in the current List. You should use your
 * reverse( ListNode * &, ListNode * & ) helper function in this method!
 *
 * @param n The size of the blocks in the List to be reversed.
 */
template <class T>
void List<T>::reverseNth(int n) {
  int group = length_ / n;
  int remain = length_ % n;
  ListNode* start = head_;
  ListNode* end = head_;

  for(int i = 0; i < group; i++) {
    for(int j = 0; j < n-1; j++) {
      end = end->next;
    }
    if(start == head_) reverse(head_,end);
    else reverse(start, end);
    start = end->next;
    end = end->next;
  }

  reverse(end,tail_);
}

/**
 * Modifies the List using the waterfall algorithm.
 * Every other node (starting from the second one) is removed from the
 * List, but appended at the back, becoming the new tail. This continues
 * until the next thing to be removed is either the tail (**not necessarily
 * the original tail!**) or NULL.  You may **NOT** allocate new ListNodes.
 * Note that since the tail should be continuously updated, some nodes will
 * be moved more than once.
 */
template <class T>
void List<T>::waterfall() {
  ListNode* newTail = head_;
  ListNode* curNext = head_;
  while(newTail && curNext->next){
    curNext = curNext->next;
    if(curNext == tail_) break;
    newTail = curNext;
    curNext = curNext->next;

    curNext->prev = newTail->prev;
    newTail->prev->next = curNext;
    newTail->prev = tail_;
    tail_->next = newTail;
    newTail->next = NULL;
    tail_ = newTail;
  }
}

/**
 * Splits the given list into two parts by dividing it at the splitPoint.
 *
 * @param splitPoint Point at which the list should be split into two.
 * @return The second list created from the split.
 */
template <class T>
List<T> List<T>::split(int splitPoint) {
    if (splitPoint > length_)
        return List<T>();

    if (splitPoint < 0)
        splitPoint = 0;

    ListNode* secondHead = split(head_, splitPoint);

    int oldLength = length_;
    if (secondHead == head_) {
        // current list is going to be empty
        head_ = NULL;
        tail_ = NULL;
        length_ = 0;
    } else {
        // set up current list
        tail_ = head_;
        while (tail_->next != NULL)
            tail_ = tail_->next;
        length_ = splitPoint;
    }

    // set up the returned list
    List<T> ret;
    ret.head_ = secondHead;
    ret.tail_ = secondHead;
    if (ret.tail_ != NULL) {
        while (ret.tail_->next != NULL)
            ret.tail_ = ret.tail_->next;
    }
    ret.length_ = oldLength - splitPoint;
    return ret;
}

/**
 * Helper function to split a sequence of linked memory at the node
 * splitPoint steps **after** start. In other words, it should disconnect
 * the sequence of linked memory after the given number of nodes, and
 * return a pointer to the starting node of the new sequence of linked
 * memory.
 *
 * This function **SHOULD NOT** create **ANY** new List objects!
 *
 * @param start The node to start from.
 * @param splitPoint The number of steps to walk before splitting.
 * @return The starting node of the sequence that was split off.
 */
template <class T>
typename List<T>::ListNode* List<T>::split(ListNode* start, int splitPoint) {
    if(start == NULL || splitPoint >= length_) return NULL;
    ListNode * temp = start;
    int count = splitPoint;
    while(count != 0){
      count--;
      temp = temp->next;
    }
    temp->prev->next = NULL;
    temp->prev = NULL;
    return temp;
}

/**
 * Merges the given sorted list into the current sorted list.
 *
 * @param otherList List to be merged into the current list.
 */
template <class T>
void List<T>::mergeWith(List<T>& otherList) {
    // set up the current list
    head_ = merge(head_, otherList.head_);
    tail_ = head_;

    // make sure there is a node in the new list
    if (tail_ != NULL) {
        while (tail_->next != NULL)
            tail_ = tail_->next;
    }
    length_ = length_ + otherList.length_;

    // empty out the parameter list
    otherList.head_ = NULL;
    otherList.tail_ = NULL;
    otherList.length_ = 0;
}

/**
 * Helper function to merge two **sorted** and **independent** sequences of
 * linked memory. The result should be a single sequence that is itself
 * sorted.d
 *
 * This function **SHOULD NOT** create **ANY** new List objects.
 *
 * @param first The starting node of the first sequence.
 * @param second The starting node of the second sequence.
 * @return The starting node of the resulting, sorted sequence.
 */
template <class T>
typename List<T>::ListNode* List<T>::merge(ListNode* first, ListNode* second) {
  if(first == NULL && second == NULL) return NULL;
  else if(first == NULL) return second;
  else if(second == NULL) return first;
  else if(first == second) return first;

  ListNode* f = first;
  ListNode* s = second;
  ListNode* temp = first;

  if(s->data < f->data){
    temp = s;
    s = s->next;
    if(s != NULL)
    s->prev = temp;
  }else{
    f = f->next;
  }
  
  ListNode* tempHead = temp;

  while(f && s){
    if(s->data < f->data){
      temp->next = s;
      s->prev = temp;
      temp = temp->next;
      s = s->next;
    } else {
      temp->next = f;
      f->prev = temp;
      temp = temp->next;
      f = f->next;
    }
  }

  if(f != NULL){
    temp->next = f;
    f->prev = temp;
  } else if(s != NULL){
    temp->next = s;
    s->prev = temp;
  }
  f = NULL;
  s = NULL;
  return tempHead;
}

/**
 * Sorts the current list by applying the Mergesort algorithm.
 */
template <class T>
void List<T>::sort() {
    if (empty())
        return;
    head_ = mergesort(head_, length_);
    tail_ = head_;
    while (tail_->next != NULL)
        tail_ = tail_->next;
}

/**
 * Sorts a chain of linked memory given a start node and a size.
 * This is the recursive helper for the Mergesort algorithm (i.e., this is
 * the divide-and-conquer step).
 *
 * @param start Starting point of the chain.
 * @param chainLength Size of the chain to be sorted.
 * @return A pointer to the beginning of the now sorted chain.
 */
template <class T>
typename List<T>::ListNode* List<T>::mergesort(ListNode* start, int chainLength) {
  if(start==NULL || start->next == NULL) return start;
  ListNode* end = split(start, chainLength/2);
  return merge(mergesort(start,chainLength/2), mergesort(end, chainLength- chainLength/2));
}

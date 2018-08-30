/**
 * @file quackfun.cpp
 * This is where you will implement the required functions for the
 *  stacks and queues portion of the lab.
 */

namespace QuackFun {

/**
 * Sums items in a stack.
 * @param s A stack holding values to sum.
 * @return The sum of all the elements in the stack, leaving the original
 *  stack in the same state (unchanged).
 *
 * @note You may modify the stack as long as you restore it to its original
 *  values.
 * @note You may use only two local variables of type T in your function.
 *  Note that this function is templatized on the stack's type, so stacks of
 *  objects overloading the + operator can be summed.
 * @note We are using the Standard Template Library (STL) stack in this
 *  problem. Its pop function works a bit differently from the stack we
 *  built. Try searching for "stl stack" to learn how to use it.
 * @hint Think recursively!
 */
template <typename T>
T sum(stack<T>& s)
{
  T local = 0, hold = 0;
  if(s.empty()) return T();
  local = s.top();
  hold = local;
  s.pop();
  hold += sum(s);
  s.push(local);
  return hold; // stub return value (0 for primitive types). Change this!
                // Note: T() is the default value for objects, and 0 for
                // primitive types
}

/**
 * Reverses even sized blocks of items in the queue. Blocks start at size
 * one and increase for each subsequent block.
 * @param q A queue of items to be scrambled
 *
 * @note Any "leftover" numbers should be handled as if their block was
 *  complete.
 * @note We are using the Standard Template Library (STL) queue in this
 *  problem. Its pop function works a bit differently from the stack we
 *  built. Try searching for "stl stack" to learn how to use it.
 * @hint You'll want to make a local stack variable.
 */
template <typename T>
void scramble(queue<T>& q)
{
    stack<T> s;
    queue<T> q2;
    int leftover = 1;
    while(!q.empty())
    {
      if(leftover%2 == 0)
      {
        for(int i = 0; i < leftover; i++)
        {
          if(q.empty()) break;
          s.push(q.front());
          q.pop();
        }
        while(!s.empty())
        {
          q2.push(s.top());
          s.pop();
        }
      }
     else{
      for(int k = 0; k < leftover; k++){
        if(q.empty()) break;
        q2.push(q.front());
        q.pop();
      }
    }
    leftover++;
  }
    while(!q2.empty())
    {
      q.push(q2.front());
      q2.pop();
    }
}

/**
 * @return true if the parameter stack and queue contain only elements of
 *  exactly the same values in exactly the same order; false, otherwise.
 *
 * @note You may assume the stack and queue contain the same number of items!
 * @note There are restrictions for writing this function.
 * - Your function may not use any loops
 * - In your function you may only declare ONE local boolean variable to use in
 *   your return statement, and you may only declare TWO local variables of
 *   parametrized type T to use however you wish.
 * - No other local variables can be used.
 * - After execution of verifySame, the stack and queue must be unchanged. Be
 *   sure to comment your code VERY well.
 */
template <typename T>
bool verifySame(stack<T>& s, queue<T>& q)
{
    bool retval = true;
    T stackTemp;
    T queueTemp;

    if(s.empty()) return retval;
    stackTemp = s.top();
    s.pop();// pop everything from stack
    retval = verifySame(s, q);// recursive case
    queueTemp = q.front();
    /**
    * compare values from stack with values from queue
    *
    * if retval is false
    * then we only need to push stackTemp back to the stack
    */
    if(retval) retval = stackTemp == queueTemp;
    s.push(stackTemp);
    q.push(q.front());
    q.pop();

    return retval;
}

}

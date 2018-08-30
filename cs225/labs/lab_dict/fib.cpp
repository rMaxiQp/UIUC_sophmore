/**
 * @file fib.cpp
 * Contains Fibonacci-number generating functions both utilizing and not
 * utilizing memoization.
 *
 * @author Matt Joras
 * @date Winter 2013
 */

#include "fib.h"

#include <map>

using std::map;

/**
 * Calculates the nth Fibonacci number where the zeroth is defined to be
 * 0.
 * @param n Which number to generate.
 * @return The nth Fibonacci number.
 */
unsigned long fib(unsigned long n)
{
  if(n == 0) return 0;
  if(n == 1) return 1;
  return fib(n-1) + fib(n-2);
}

/**
 * Calculates the nth Fibonacci number where the zeroth is defined to be
 * 0. This version utilizes memoization.
 * @param n Which number to generate.
 * @return The nth Fibonacci number.
 */
unsigned long memoized_fib(unsigned long n)
{
  static map<unsigned long, unsigned long> dp = {
    {0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 3}
  };
  map<unsigned long, unsigned long>::iterator it = dp.find(n);
  if(it != dp.end())
  {
    return it->second;
  }
  else
  {
    unsigned long result = memoized_fib(n-2) + memoized_fib(n-1);
    dp[n] = result;
    return result;
  }
}

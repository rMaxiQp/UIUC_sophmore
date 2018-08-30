/**
 * @file cartalk_puzzle.cpp
 * Holds the function which solves a CarTalk puzzler.
 *
 * @author Matt Joras
 * @date Winter 2013
 */

#include <fstream>
#include "cartalk_puzzle.h"
#include <iostream>
using namespace std;

/**
 * Solves the CarTalk puzzler described here:
 * http://www.cartalk.com/content/wordplay-anyone.
 * @return A vector of (string, string, string) tuples
 * Returns an empty vector if no solutions are found.
 * @param d The PronounceDict to be used to solve the puzzle.
 * @param word_list_fname The filename of the word list to be used.
 */
vector<std::tuple<std::string, std::string, std::string>> cartalk_puzzle(PronounceDict d,
                                    const string& word_list_fname)
{
  vector<std::tuple<std::string, std::string, std::string>> ret;

  vector<string> ss;
  ifstream wordsFile(word_list_fname);
  string word;
  if (wordsFile.is_open())
  {
    while (getline(wordsFile, word))
    {
      if(word.length() == 5) ss.push_back(word);
    }
  }

  for(auto a : ss)
  {
    string s0 = a;
    string s1 = a.substr(1);
    string s2 = a[0] + a.substr(2);
    if(d.homophones(s0,s1) && d.homophones(s1,s2))
    {
      ret.push_back({s0, s1, s2});
    }
  }

  return ret;
}

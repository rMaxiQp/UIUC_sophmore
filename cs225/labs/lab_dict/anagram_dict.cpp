/**
 * @file anagram_dict.cpp
 * Implementation of the AnagramDict class.
 *
 * @author Matt Joras
 * @date Winter 2013
 */

#include "anagram_dict.h"
#include <iostream>
#include <algorithm> /* I wonder why this is included... */
#include <fstream>

using std::string;
using std::vector;
using std::ifstream;

/**
 * Constructs an AnagramDict from a filename with newline-separated
 * words.
 * @param filename The name of the word list file.
 */
AnagramDict::AnagramDict(const string& filename)
{
  ifstream wordsFile(filename);
  string word;
  if (wordsFile.is_open())
  {
    while (getline(wordsFile, word))
    {
      vector<string> v;
      string temp = word;
      sort(temp.begin(), temp.end());
      auto it = dict.find(temp);
      if(it != dict.end())
      {
        bool flag = true;
        for(size_t t = 0; t < it->second.size(); t++)
        {
          if(it->second[t] == word)
          {
            flag = false;
            break;
          }
        }
        if(flag){
          it->second.push_back(word);
        }
      }
      else
      {
        dict[temp].push_back(word);
      }
    }
  }
}

/**
 * Constructs an AnagramDict from a vector of words.
 * @param words The vector of strings to be used as source words.
 */
AnagramDict::AnagramDict(const vector<string>& words)
{
  for(auto a : words)
  {
    vector<string> v;
    string temp = a;
    sort(temp.begin(), temp.end());
    auto it = dict.find(temp);
    if(it != dict.end())
    {
      bool flag = true;
      for(size_t t = 0; t < it->second.size(); t++)
      {
        if(it->second[t] == a)
        {
          flag = false;
          break;
        }
      }
      if(flag){
        it->second.push_back(a);
      }
    }
    else
    {
      dict[temp].push_back(a);
    }
  }
}

/**
 * @param word The word to find anagrams of.
 * @return A vector of strings of anagrams of the given word. Empty
 * vector returned if no anagrams are found or the word is not in the
 * word list.
 */
vector<string> AnagramDict::get_anagrams(const string& word) const
{
  string temp = word;
  sort(temp.begin(), temp.end());
  auto a = dict.find(temp);
  vector<string> v;
  if(a != dict.end() && a->second.size() > 1)
  {
    for(size_t t = 0; t < a->second.size(); t++)
    {
      v.push_back(a->second[t]);
    }
  }
  return v;
}

/**
 * @return A vector of vectors of strings. Each inner vector contains
 * the "anagram siblings", i.e. words that are anagrams of one another.
 * NOTE: It is impossible to have one of these vectors have less than
 * two elements, i.e. words with no anagrams are ommitted.
 */
vector<vector<string>> AnagramDict::get_all_anagrams() const
{
  vector<vector<string>> v;
    for(auto &a : dict)
    {
      vector<string> temp = get_anagrams(a.first);
      if(temp != vector<string>()) v.push_back(temp);
    }
    return v;
}

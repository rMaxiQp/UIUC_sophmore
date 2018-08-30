#include "Pattern.h"
#include <string>
#include <iostream>
#include <vector>

bool wordPattern(std::string pattern, std::string str)
{
  vector<string> holder;
  vector<string> potential;
  string k = "";
  cout<<pattern<<endl;
  cout<<str<<endl;
  for(char c: str)
  {
    if(c == ' ')
    {
      potential.push_back(k);
      k = "";
    }
    else
    {
      k = k + c;
    }
  }
  potential.push_back(k);
  for(int i = 0; i < 26; i++)
  {
    holder.push_back("-1");
  }
  int i = 0;
  for(char c : pattern)
  {
    int idx = c - 'a';
    cout<<idx<<endl;
    if(idx != 0 && holder.at(0) == potential.at(i)) return false;
    if(holder.at(idx) == "-1") holder.at(idx) = potential.at(i);
    else if(holder.at(idx) != potential.at(i)) return false;
    i++;
  }
  cout<<holder.at(0)<<endl;
  cout<<holder.at(1)<<endl;
  return true;
}

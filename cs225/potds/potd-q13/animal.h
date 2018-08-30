#ifndef _ANIMAL_H
#define _ANIMAL_H

#include <string>
#include <iostream>

using namespace std;
// namespace potd{
  class Animal{
    public:
      Animal();
      Animal(string t, string f);
      string getType();
      void setType(string t);
      string getFood();
      void setFood(string f);
      string print();

    private:
      string type_;
      string food_;
  };

#endif

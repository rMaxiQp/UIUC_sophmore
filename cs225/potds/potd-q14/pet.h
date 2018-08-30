#ifndef _PET_H
#define _PET_H

#include "animal.h"

class Pet : public Animal{
public:
  string getFood();
  string getType();
  string getOwnerName();
  string getName();
  void setName(string n);
  void setFood(string f);
  void setType(string t);
  void setOwnerName(string o);
  string print();
  Pet();
  Pet(string t, string f, string n, string o);
private:
  string type_;
  string food_;
  string name_;
  string owner_name_;
};

#endif

#ifndef FOOD_H
#define FOOD_H

#include <string>

using namespace std;

class food{
public:
  food();
  string get_name();
  void set_name(string newName);
  unsigned get_quantity();
  void set_quantity(unsigned newQuantity);
private:
  string name_;
  unsigned quantity_;
};

#endif

#include <string>

#include "food.h"

food::food(){
  name_ = "null";
  quantity_ = 0;
}

string food::get_name(){
  return name_;
}

void food::set_name(string newName){
  name_ = newName;
}

unsigned food::get_quantity(){
  return quantity_;
}

void food::set_quantity(unsigned newQuantity){
  quantity_ = newQuantity;
}

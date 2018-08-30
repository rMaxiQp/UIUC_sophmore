#include "animal.h"

//using namespace potd;

string Animal::getType(){
  return type_;
}

string Animal::getFood(){
  return food_;
}

void Animal::setType(string t){
  type_ = t;
}

void Animal::setFood(string f){
  food_ = f;
}

string Animal::print(){
  string temp = "I am a " + getType();
  return temp;
}

Animal::Animal() : Animal::Animal("cat", "fish"){}

Animal::Animal(string t, string f){
  type_ = t;
  food_ = f;
}

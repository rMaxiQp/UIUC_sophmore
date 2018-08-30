#include "pet.h"

string Pet::getFood(){return food_;}
string Pet::getType(){return type_;}
string Pet::getOwnerName(){return owner_name_;}
string Pet::getName(){return name_;}
void Pet::setName(string n){
  name_ = n;
}
void Pet::setFood(string f){
  food_ = f;
}
void Pet::setType(string t){
  type_ = t;
}
void Pet::setOwnerName(string o){
  owner_name_ = o;
}

string Pet::print(){
  string temp = "My name is " + name_;
  return temp;
}
Pet::Pet(string t, string f, string n, string o){
  type_ = t;
  food_ = f;
  name_ = n;
  owner_name_ = o;
}
Pet::Pet(): type_("cat"), food_("fish"), name_("Fluffy"), owner_name_("Cinda"){}

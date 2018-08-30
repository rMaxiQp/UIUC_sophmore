#include "thing.h"
#include <iostream>

using namespace potd;
using namespace std;

Thing::Thing(int size){
  props_ct_ = 0;
  props_max_ = size;
  properties_ = new string[size];
  values_ = new string[size];
}

Thing::Thing(const Thing &t){
  _copy(t);
}

void Thing::_copy(const Thing &t){
  this->props_max_ = t.props_max_;
  this->props_ct_ = t.props_ct_;
  this->properties_ = new string[props_max_];
  this->values_ = new string[props_max_];
  for(int i = 0; i< props_ct_; i++){
    properties_[i] = t.properties_[i];
    values_[i] = t.values_[i];
  }
}

Thing & Thing::operator=(const Thing &t){
  if (this == &t) return *this;
  delete[] properties_;
  delete[] values_;
  _copy(t);
  return *this;
}

int Thing::set_property(string name, string value){
  for(int i = 0; i < props_ct_; i++){
    if(properties_[i] == name){
      values_[i] = value;
      return i;
    }
  }
  if(props_ct_ == props_max_) return -1;
  properties_[props_ct_] = name;
  values_[props_ct_] = value;
  props_ct_ ++;
  return props_ct_-1;
}

string Thing::get_property(string name){
  int i = 0;
  while(i < props_ct_){
    if(properties_[i] == name)
      return values_[i];
    i++;
  }
  return "";
}

void Thing::_destroy(){
  Thing::~Thing();
}

Thing::~Thing(){
  delete[] properties_;
  delete[] values_;
}

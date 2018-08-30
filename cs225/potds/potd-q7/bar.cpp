#include "bar.h"

string bar::get_name(){ return f_->get_name();}

bar::bar(string newString){ f_ = new foo::foo(newString);}

bar::bar(const bar& that){
  f_ =  new foo(*that.f_);
}

bar & bar::operator=(const potd::bar &that){
  delete f_;
  this->f_ = new foo(*that.f_);
  return *this;
}

bar::~bar(){ f_->~foo();}

#include <iostream>

#include "q6.h"
#include "student.h"

using namespace potd;

int main(){
  student s;
  cout << s.get_grade()<< endl;
  graduate(s);
  cout << s.get_grade()<< endl;
}

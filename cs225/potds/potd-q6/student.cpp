#include "student.h"

using namespace potd;

student::student(){
  name_ = "BLAH";
  grade_ = 10;
}

string student::get_name(){ return name_;}

void student::set_name(string newName){ name_ = newName;}

void student::set_grade(unsigned newGrade){grade_ = newGrade;}

unsigned student::get_grade(){return grade_;}

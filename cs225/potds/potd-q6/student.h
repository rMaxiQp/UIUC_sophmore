#ifndef STUDENT_H
#define STUDENT_H
#include <string>

using namespace std;
namespace potd{
  class student{
  public:
    student();
    void set_name(string newName);
    string get_name();
    void set_grade(unsigned newGrade);
    unsigned get_grade();
  private:
    string name_;
    unsigned grade_;
  };
}
#endif

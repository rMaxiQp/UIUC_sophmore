#include <iostream>
#include <string>
#include "hello.h"

std::string hello(){
  std::string your_name = "Max Qian";
  std::string your_age = "20";
  std::string temp = "Hello world! My name is " + your_name + " and I am " + your_age + " years old.";
  return temp;
}

#include <iostream>
#include <string>
using namespace std;

#include "square.h"

Square::Square() {
    name = "mysquare";
    lengthptr = new double;
    *lengthptr = 2.0;
}

void Square::setName(string newName) {
  name = newName;
}

void Square::setLength(double newVal) {
  *lengthptr = newVal;
}

string Square::getName() const {
  return name;
}

double Square::getLength() const {
  return *lengthptr;
}

Square::Square(const Square & other) {
    name = other.getName();
    lengthptr = new double;
    *lengthptr = other.getLength();
}

Square & Square::operator=(const Square & other){
  name = other.getName();
  lengthptr = new double;
  *lengthptr = other.getLength();
  return *this;
}

Square Square::operator+(const Square & other){
  string name_old = getName();
  double length_old = getLength();
  Square square_new;
  square_new.setName(name_old + other.getName());
  square_new.setLength(length_old + other.getLength());
  return square_new;
}


Square::~Square() {
    delete lengthptr;
}

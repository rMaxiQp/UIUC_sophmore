#include "potd.h"
string getFortune(const string &s){
  int choose = s.length() % 5;
  switch(choose){
    case 0: return "As you wish!";
    case 1: return "Nec spe nec metu!";
    case 2: return "Do, or do not. There is no try.";
    case 3: return "This fortune intentionally left blank.";
    default: return "Amor Fati!";
  }
}

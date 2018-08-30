#ifndef _THING_2
#define _THING_2
#include <string>
#include "thing1.h"
using namespace std;

class Thing2 : public Thing1{
public:
  string foo();
  virtual string bar();
  virtual ~Thing2();
};

#endif

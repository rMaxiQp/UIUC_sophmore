#ifndef BAR_H
#define BAR_H

#include <string>
#include "foo.h"

using namespace std;
using namespace potd;

namespace potd{
class bar{
public:
  bar(string newString);
  ~bar();
  bar(const bar& that);
  bar & operator= (const bar &);
  string get_name();
private:
  foo *f_;
};
}

#endif

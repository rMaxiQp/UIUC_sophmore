#ifndef THING_H
#define THING_H

#include <string>

namespace potd {
  class Thing {
  public:
    Thing(int);
    Thing(const Thing &);
    Thing & operator=(const Thing &);
    ~Thing();
    int props_ct_;
    int set_property(std::string,std::string);
    std::string get_property(std::string);

  private:

    void _copy(const Thing &);
    void _destroy();

    std::string *properties_;
    std::string *values_;

    int props_max_;
  };
}
#endif

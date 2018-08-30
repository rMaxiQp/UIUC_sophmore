#include "HSLAPixel.h"
using namespace cs225;

//A default pixel is completely opaque and white.
  HSLAPixel::HSLAPixel(){
    h = 0.0;
    s = 0.0;
    l = 1.0;
    a = 1.0;
  }

//Constructs an opaque HSLAPixel with the given hue, saturation, and luminance values.
  HSLAPixel::HSLAPixel(double hue, double saturation, double luminance){
    h = hue;
    s = saturation;
    l = luminance;
    a = 1.0;
  }

//Constructs an opaque HSLAPixel with the given hue, saturation, luminance, and alpha values.
  HSLAPixel::HSLAPixel(double hue, double saturation, double luminance, double alpha){
    h = hue;
    s = saturation;
    l = luminance;
    a = alpha;
  }

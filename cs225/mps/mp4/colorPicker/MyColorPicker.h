#ifndef MyColorPicker_H
#define MyColorPicker_H

#include "ColorPicker.h"
#include "../cs225/HSLAPixel.h"
#include "../Point.h"

using namespace cs225;

class MyColorPicker : public ColorPicker {
public:
  HSLAPixel getColor(unsigned x, unsigned y);

private:
  HSLAPixel blue = HSLAPixel(240.0, 1.0, 0.5);
  HSLAPixel orange = HSLAPixel(160.0, 1.0, 0.5);
};

#endif

#include "../cs225/HSLAPixel.h"
#include "../Point.h"

#include "ColorPicker.h"
#include "MyColorPicker.h"

using namespace cs225;

/**
 * Picks the color for pixel (x, y).
 */
HSLAPixel MyColorPicker::getColor(unsigned x, unsigned y) {
  if(x > y + 100 || y > x + 100) return orange;
  if(x > y || y < x ) return blue;
  return HSLAPixel();
}

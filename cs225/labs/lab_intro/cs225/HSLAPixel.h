#ifndef HSLAPIXEL_H
#define HSLAPIXEL_H

namespace cs225{
  class HSLAPixel{
  public:
    double h;//Double for the hue of the pixel, in degrees [0, 360].
    double s;//Double for the saturation of the pixel, [0, 1].
    double l;//Double for the luminance of the pixel, [0, 1].
    double a;//Double for the alpha of the pixel, [0, 1].

    HSLAPixel();

    HSLAPixel(double hue, double saturation, double luminance);

    HSLAPixel(double hue, double saturation, double luminance, double alpha);
  };
}

#endif

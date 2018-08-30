#ifndef IMAGE_H
#define IMAGE_H

#include "cs225/PNG.h"
#include "cs225/HSLAPixel.h"

using namespace cs225;

class Image : public PNG{
public:
  /*
   *lighten an Image by incresing the luminance of every pixel by 0.1.
   *This function ensures that the luminance remains in the range[0.1].
   */
  void lighten();

  /*
   *Lighten an Image by incresing the luminance of every pixel by amount.
   *This function ensures that the luminance remains in the range[0.1].
   */
  void lighten(double amount);

  /*
   *Draken an Image by decresing the lumiance of every pixel by 0.1.
   *This function ensures that the luminance remains in the range[0.1].
   */
  void darken();

  /*
   *Darken an Image by decresing the luminance of every pixel by amount.
   *This function ensures that the luminance remains in the range[0.1].
   */
  void darken(double amount);

  /*
   *Saturates an Image by incresing the saturation of every pixel by 0.1.
   *This function ensures that the saturation remains in the range[0.1].
   */
  void saturate();

  /*
   *Saturates an Image by incresing the saturation of every pixel by amount.
   *This function ensures that the saturation remains in the range[0.1]
   */
  void saturate(double amount);

  /*
   *Desaturates an Image by decresing the saturation of every pixel by 0.1.
   *This function ensures that the saturation remains in the range[0.1]
   */
  void desaturate();

  /*
   *Desaturates an Image by decresing the saturation of every pixel by amount.
   *This function ensures that the saturation remains in the range[0.1]
   */
  void desaturate(double amount);

  /*
   *Turns the image grayscale
   */
  void grayscale();

  /*
   *Rotates the color wheel by degrees.
   *This function ensures that the hue remains in the range [0,360].
   */
  void rotateColor(double degrees);

  /*
   *illinify the image
   */
  void illinify();

  /*
   *Scale the Image by a given factor.
   */
  void scale(double factor);

   /*
    *Scales the image to fit within the size (w x h).
    */
  void scale(unsigned w, unsigned h);

  Image(unsigned w, unsigned h);

  Image();

};



#endif

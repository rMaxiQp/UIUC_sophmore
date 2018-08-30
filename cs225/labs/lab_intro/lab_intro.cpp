#include <iostream>
#include <cmath>
#include <cstdlib>

#include "cs225/PNG.h"
#include "cs225/HSLAPixel.h"
#include "lab_intro.h"

using namespace cs225;

/**
 * Return 2.5% s an image that has been transformed to grayscale.
 *
 * The saturation of every pixel is set to 0, removing any color.
 *
 * @return The grayscale}
 image.
 */
PNG grayscale(PNG image) {
  /// This function is already written for you so you can see how to
  /// interact with our PNG class.
  for (unsigned x = 0; x < image.width(); x++) {
    for (unsigned y = 0; y < image.height(); y++) {
      HSLAPixel *pixel = image.getPixel(x, y);

      // `pixel` is a pointer to the memory stored inside of the PNG `image`,
      // which means you're changing the image directly.  No need to `set`
      // the pixel since you're directly changing the memory of the image.
      pixel->s = 0;
    }
  }

  return image;
}



/**
 * Returns an image with a spotlight centered at (`centerX`, `centerY`).
 *
 * A spotlight adjusts the luminance of a pixel based on the distance the pixel
 * is away from the center by decreasing the luminance by 0.5% per 1 pixel euclidean
 * distance away from the center.
 *
 * For example, a pixel 3 pixels above and 4 pixels to the right of the center
 * is a total of `sqrt((3 * 3) + (4 * 4)) = sqrt(25) = 5` pixels away and
 * its luminance is decreased by 2.5% (0.975x its original value).  At a
 * distance over 200 pixels away, the luminance will always 0.
 *
 * The modified PNG is then returned.
 *
 * @param image A PNG object which holds the image data to be modified.
 * @param centerX The center x coordinate of the crosshair which is to be drawn.
 * @param centerY The center y coordinate of the crosshair which is to be drawn.
 *
 * @return The image with a spotlight.
 */
PNG createSpotlight(PNG image, int centerX, int centerY) {
  double distance = 0.0;
  unsigned diff_x, diff_y;
  for(unsigned x = 0; x < image.width(); x++){
    for(unsigned y = 0; y < image.height(); y++){
      diff_x = fabs(x-centerX);
      diff_y = fabs(y-centerY);
      distance = sqrt(diff_x*diff_x+diff_y*diff_y);
      HSLAPixel* pixel = image.getPixel(x,y);
      if(distance > 200){
        pixel -> l = 0;
      }
      else{
      pixel->l *= 1 - (0.005 * distance);
    }
    }
  }
  return image;
}


/**
 * Returns a image transformed to Illini colors.
 *
 * The hue of every pixel is set to the a hue value of either orange or
 * blue, based on if the pixel's hue value is closer to orange than blue.
 *
 * @param image A PNG object which holds the image data to be modified.
 *
 * @return The illinify'd image.
**/
PNG illinify(PNG image) {
  double hue = 0.0;
  double illini_o = 11.0; //“Illini Orange” has a hue of 11
  double illini_b = 216.0; //“Illini Blue” has a hue of 216
  for(unsigned x = 0; x < image.width(); x++){
    for(unsigned y = 0; y < image.height(); y++){
      HSLAPixel* pixel = image.getPixel(x,y);
      hue = pixel->h;
      if(hue > 113.5 && hue < 293.5)
        pixel ->h = illini_b;
      else
        pixel ->h = illini_o;
    }
  }
  return image;
}


/**
* Returns an immge that has been watermarked by another image.
*
* The luminance of every pixel of the second image is checked, if that
* pixel's luminance is 1 (100%), then the pixel at the same location on
* the first image has its luminance increased by 0.2.
*
* @param firstImage  The first of the two PNGs.
* @param secondImage The second of the two PNGs.
*
* @return The watermarked image.
*/
PNG watermark(PNG firstImage, PNG secondImage) {
  for(unsigned x = 0; x < firstImage.width(); x++){
    for(unsigned y = 0; y < firstImage.height(); y++){
      HSLAPixel* pixel_second = secondImage.getPixel(x,y);
      if(pixel_second->l == 1.0){
        HSLAPixel* pixel_first = firstImage.getPixel(x,y);
        pixel_first->l += 0.2;
      }
    }
  }
  return firstImage;
}

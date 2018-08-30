#include "Image.h"
#include <cmath>
Image::Image(){}

Image::Image(unsigned w, unsigned h) : PNG(w,h){}

void Image::lighten(){
  Image::lighten(0.1);
}

void Image::lighten(double amount){
  unsigned w = width();
  unsigned h = height();
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    if(pixel->l+amount <= 1.0) pixel->l += amount;
    else pixel->l = 1.0;
  }
}

void Image::darken(){
  Image::darken(0.1);
}

void Image::darken(double amount){
  unsigned w = width();
  unsigned h = height();
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    if(pixel->l > amount) pixel->l -= amount;
    else pixel->l = 0.0;
  }
}

void Image::saturate(){
  Image::saturate(0.1);
}

void Image::saturate(double amount){
  unsigned w = width();
  unsigned h = height();
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    if(amount + pixel->s <= 1.0) pixel->s += amount;
    else pixel->s = 1.0;
  }
}

void Image::desaturate(){
  Image::desaturate(0.1);
}

void Image::desaturate(double amount){
  unsigned w = width();
  unsigned h = height();
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    if(pixel->s > amount) pixel->s -= amount;
    else pixel->s = 0.0;
  }
}

void Image::grayscale(){
  unsigned w = width();
  unsigned h = height();
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    pixel->s = 0;
  }
}

void Image::rotateColor(double degrees){
  unsigned w = width();
  unsigned h = height();
  degrees = fmod(degrees, 360.0);
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    pixel->h += degrees;
    if(pixel->h > 360.0) pixel->h -= 360.0;
    else if(pixel->h < 0.0) pixel->h += 360.0;
  }
}

void Image::illinify(){
  unsigned w = width();
  unsigned h = height();
  double illini_o = 11.0; //“Illini Orange” has a hue of 11
  double illini_b = 216.0; //“Illini Blue” has a hue of 216
  for(unsigned i = 0; i < w*h; i++){
    HSLAPixel* pixel = getPixel(i%w,i/w);
    if(pixel->h > 113.5 && pixel->h < 293.5)
      pixel->h = illini_b;
    else
      pixel->h = illini_o;
 }
}

void Image::scale(double factor){
  unsigned newWidth = width() *1.0 * factor;
  unsigned newHeight = height() *1.0* factor;
  Image temp = Image(newWidth, newHeight);
  for(unsigned i = 0; i < newWidth; i++){
    for(unsigned j = 0; j < newHeight; j++){
      HSLAPixel *newPixel = temp.getPixel(i,j);
      HSLAPixel *pixel = getPixel(i/factor*1,j/factor*1);
      newPixel->l = pixel->l;
      newPixel->s = pixel->s;
      newPixel->h = pixel->h;
      newPixel->a = pixel->a;
    }
  }
  *this = temp;
}

void Image::scale(unsigned w, unsigned h){
  double factor_w = (w/1.0) /width();
  double factor_h = (h/1.0) /height();
  if (factor_w < factor_h)
    Image::scale(factor_w);
  else
    Image::scale(factor_h);
}

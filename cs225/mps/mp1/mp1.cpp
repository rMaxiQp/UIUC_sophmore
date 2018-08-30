#include <string>

#include "cs225/PNG.h"
#include "mp1.h"
#include "cs225/HSLAPixel.h"

using namespace cs225;

void rotate(std::string inputFile, std::string outputFile) {
  PNG input = PNG();
  input.readFromFile(inputFile);
  unsigned width = input.width();
  unsigned height = input.height();
  PNG output = PNG(width, height);

  for(unsigned x = 0; x < width; x++){
    for(unsigned y = 0; y < height; y++){
      HSLAPixel *pixel_in = input.getPixel(x,y);
      HSLAPixel *pixel_out = output.getPixel(width-x-1, height-y-1);
      pixel_out->h = pixel_in->h;
      pixel_out->s = pixel_in->s;
      pixel_out->l = pixel_in->l;
      pixel_out->a = pixel_in->a;
    }
  }

  output.writeToFile(outputFile);

  return;
}//end of rotate()

#include "StickerSheet.h"
#include <string>
using namespace cs225;
using namespace std;

int main() {
  Image pf, w, m, logo, res;
  pf.readFromFile("pfic.png");
  w.readFromFile("wade.png");
  logo.readFromFile("M.png");
  m.readFromFile("mattox.png");
  StickerSheet *sticker = new StickerSheet(pf, 5);
  w.scale(0.6);
  m.scale(0.6);
  logo.scale(0.22);

  sticker->addSticker(w, 580, 88);
  sticker->addSticker(m, 355, 66);
  sticker->addSticker(logo, 10, 300);

  res = sticker->render();

  res.writeToFile("myImage.png");

  delete sticker;
  return 0;
}

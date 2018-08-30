#include "StickerSheet.h"
#include <string>
using namespace cs225;
using namespace std;

int main() {
  Image alma, UofI,res, s;
  alma.readFromFile("alma.png");
  UofI.readFromFile("i.png");
  s.readFromFile("i.png");
  StickerSheet *sticker = new StickerSheet(alma, 10);
  sticker->addSticker(UofI, 0, 0);
  sticker->addSticker(s, 20, 10);

  res = sticker->render();
  res.writeToFile("o.png");

  delete sticker;
  return 0;
}

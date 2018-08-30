#include "StickerSheet.h"
#include <string>
using namespace cs225;
using namespace std;

int main() {
  Image alma, UofI, res;
  alma.readFromFile("alma.png");
  UofI.readFromFile("i.png");

  StickerSheet* sticker = new StickerSheet(alma, 3);
  StickerSheet* s = new StickerSheet(alma, 5);
  sticker->addSticker(UofI, 0, 0);
  sticker->addSticker(UofI, 20, 10);

  //getSticker
  Image* temp = sticker->getSticker(1);
  cout<< temp << " at 1" <<endl;
  temp = sticker->getSticker(2);
  std::cout << "should be NULL " << &temp << '\n';

  //removeSticker
  // sticker.removeSticker(1);
  // sticker.removeSticker(2);
  //
  // sticker.removeSticker(0);
  temp = sticker->getSticker(0);

  //addSticker
  sticker->addSticker(UofI,11,22);
  sticker->addSticker(UofI,91,42);
  sticker->addSticker(UofI,31,82);
  temp = sticker->getSticker(0);

  //changeMax
  sticker->changeMaxStickers(5);
  sticker->addSticker(UofI, 2, 600);
  sticker->addSticker(UofI,900,202);
  sticker->changeMaxStickers(3);
  res = sticker->render();
  res.writeToFile("o.png");

  delete sticker;
  delete s;
  return 0;
}

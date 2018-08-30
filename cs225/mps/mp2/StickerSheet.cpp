#include "StickerSheet.h"


StickerSheet::StickerSheet(const Image &picture, unsigned max){
  count_ = 0;
  max_ = max;
  sheet_ = new Image*[max_];
  sheet_x_ = new unsigned[max_];
  sheet_y_ = new unsigned[max_];
  for(unsigned i =0; i < max_ ; i++){
    sheet_[i] = NULL;
    sheet_x_[i] = 0;
    sheet_y_[i] = 0;
  }
  base_ = new Image(picture);
}

StickerSheet::StickerSheet(const StickerSheet &other){
  _copy(other);
}

const StickerSheet & StickerSheet::operator = (const StickerSheet &other){
if(this != &other){
  _destroy();
  _copy(other);
}
  return *this;
}

void StickerSheet::changeMaxStickers(unsigned max){
  if(max == max_) return;
  StickerSheet other = StickerSheet(*base_, max);
  unsigned temp = max > max_ ? max_ : max;
  for(unsigned i = 0; i < temp; i++){

    if(sheet_[i] != NULL){
      (other.sheet_[i]) = new Image(*sheet_[i]);
      other.sheet_x_[i] = sheet_x_[i];
      other.sheet_y_[i] = sheet_y_[i];
      other.count_++;
    }
  }
  *this = other;
}

int StickerSheet::addSticker(Image &sticker, unsigned x, unsigned y){
  if(count_ == max_) return -1;
  unsigned cur = 0;
  while(sheet_[cur]!= NULL) { cur++; }
  sheet_[cur] = new Image(sticker);
  sheet_x_[cur] = x;
  sheet_y_[cur] = y;
  return count_++;
}

bool StickerSheet::translate(unsigned index, unsigned x, unsigned y){
  if(index > max_ || sheet_[index] == NULL) return false;
  sheet_x_[index] = x;
  sheet_y_[index] = y;
  return true;
}

void StickerSheet::removeSticker(unsigned index){
  if(index < max_ && sheet_[index] != NULL){
    delete sheet_[index];
    sheet_[index] = NULL;
    sheet_x_[index] = 0;
    sheet_y_[index] = 0;
    count_--;
  }
}

Image StickerSheet::render() const {
  if(count_ <= 0) return *base_;
  Image render_image = Image(*base_);
  unsigned max_height = base_->height();
  unsigned max_width = base_->width();
  if(count_ > 0){
    unsigned j = 0;
    while(j < max_ && sheet_[j] != NULL){
      if((sheet_[j]->height() + sheet_y_[j]) > max_height) max_height = sheet_[j]->height() + sheet_y_[j];
      if((sheet_[j]->width() + sheet_x_[j]) > max_width) max_width = sheet_[j]->width() + sheet_x_[j];
       j++;
    }
  }
  render_image.resize(max_width, max_height);
  for(unsigned i = 0; i < max_; i++){
    if(sheet_[i] != NULL){
      unsigned w = sheet_[i]->width();
      unsigned h = sheet_[i]->height();
      for(unsigned a = sheet_x_[i]; a < (w+sheet_x_[i]); a++){
        for(unsigned b = sheet_y_[i]; b < (h+sheet_y_[i]); b++){
          HSLAPixel * pixel = render_image.getPixel(a,b);
          HSLAPixel * old = sheet_[i]->getPixel((a-sheet_x_[i]),(b-sheet_y_[i]));
          if(old->a != 0.0) *pixel = *old;
      }
    }
    }
  }
  return render_image;
}

Image* StickerSheet::getSticker(unsigned index) const {
  if(index < max_ && sheet_[index] != NULL) return sheet_[index];
  return NULL;
}

void StickerSheet::_copy(const StickerSheet &other){
  max_ = other.max_;
  count_ = other.count_;
  base_ = new Image(*other.base_);
  sheet_ = new Image*[max_];
  sheet_x_ = new unsigned[max_];
  sheet_y_ = new unsigned[max_];
  for(unsigned i = 0; i < max_; i++){
    sheet_[i] = NULL;
    if(other.sheet_[i] != NULL){
      sheet_[i] = new Image();
      *sheet_[i] = *other.sheet_[i];
      sheet_x_[i] = other.sheet_x_[i];
      sheet_y_[i] = other.sheet_y_[i];
    }
  }
}

void StickerSheet::_destroy(){
  for(unsigned i = 0; i < max_; i++){
    if(sheet_[i] != NULL)
      delete sheet_[i];
      sheet_[i] = NULL;
  }
  delete base_;
  delete[] sheet_;
  delete[] sheet_x_;
  delete[] sheet_y_;
}

StickerSheet::~StickerSheet(){
  _destroy();
}

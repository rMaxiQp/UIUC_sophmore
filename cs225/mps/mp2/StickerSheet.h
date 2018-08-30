#ifndef _STICKERSHEET_H
#define _STICKERSHEET_H

#include "Image.h"

  class StickerSheet : public Image{
  public:
    /**Initializes this StickerSheet with a base picture and the ability to
      hold a max number of stickers (Images) with indices 0 through max - 1.
    */
    StickerSheet(const Image &picture, unsigned max);

    //Frees all space that was dynamically allocated by this StickerSheet.
    ~StickerSheet();

    //The copy constructor makes this StickerSheet an independent copy of the source.
    StickerSheet(const StickerSheet &other);

    //The assignment operator for the StickerSheet class.
    const StickerSheet & 	operator= (const StickerSheet &other);

    /**Modifies the maximum number of stickers that can be stored
      on this StickerSheet without changing existing stickers' indices.
    */
    void changeMaxStickers(unsigned max);

    /**
    Adds a sticker to the StickerSheet, so that the top-left of the sticker's Image
    is at (x, y) on the StickerSheet.
    The sticker must be added to the lowest possible layer available.
    */
    int addSticker(Image &sticker, unsigned x, unsigned y);

    //Changes the x and y coordinates of the Image in the specified layer.
    bool translate(unsigned index, unsigned x, unsigned y);

    //Removes the sticker at the given zero-based layer index.
    void removeSticker(unsigned index);

    //Returns a pointer to the sticker at the specified index, not a copy of it.
    Image* getSticker(unsigned index) const;

    //Renders the whole StickerSheet on one Image and returns that Image.
    Image render() const;

  private:
    void _destroy();
    void _copy(const StickerSheet &other);
    unsigned max_;
    unsigned count_;
    Image* base_;
    Image** sheet_;
    unsigned* sheet_x_;
    unsigned* sheet_y_;
  };

#endif

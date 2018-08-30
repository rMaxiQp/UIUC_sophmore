#include <cmath>
#include <iterator>
#include <iostream>

#include "../cs225/HSLAPixel.h"
#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"

/**
 * Calculates a metric for the difference between two pixels, used to
 * calculate if a pixel is within a tolerance.
 *
 * @param p1 First pixel
 * @param p2 Second pixel
 */
double ImageTraversal::calculateDelta(const HSLAPixel & p1, const HSLAPixel & p2) {
  double h = fabs(p1.h - p2.h);
  double s = p1.s - p2.s;
  double l = p1.l - p2.l;

  // Handle the case where we found the bigger angle between two hues:
  if (h > 180) { h = 360 - h; }
  h /= 360;

  return sqrt( (h*h) + (s*s) + (l*l) );
}


void ImageTraversal::Iterator::move(){
  //RIGHT
  if(curX+1 < width_)
  {
    Point right(curX+1, curY);
    if(!visited_[right.x][right.y] && calculateDelta(*png_.getPixel(start_.x,start_.y),*png_.getPixel(right.x,right.y))<tolerance_)
    {
      it_->add(right);
    }
  }
  //BELOW
  if(curY+1 < height_)
  {
    Point below(curX, curY+1);
    if(!visited_[below.x][below.y] && calculateDelta(*png_.getPixel(start_.x,start_.y),*png_.getPixel(below.x,below.y))<tolerance_)
    {
      it_->add(below);
    }
  }
  //LEFT
  if(curX != 0)
  {
    Point left(curX-1,curY);
    if(!visited_[left.x][left.y] && calculateDelta(*png_.getPixel(start_.x,start_.y),*png_.getPixel(left.x,left.y))<tolerance_)
    {
      it_->add(left);
    }
  }
  //ABOVE
  if(curY != 0)
  {
    Point above(curX, curY-1);
    if(!visited_[above.x][above.y] && calculateDelta(*png_.getPixel(start_.x,start_.y),*png_.getPixel(above.x,above.y))<tolerance_)
    {
      it_->add(above);
    }
  }
}


/**
 * Default iterator constructor.
 */
ImageTraversal::Iterator::Iterator() {
  it_ = NULL;
  start_ = Point(-1,-1);
  curX = -1;
  curY = -1;
  png_ = PNG();
  tolerance_ = -1.0;
  width_ = 0;
  height_ = 0;
}

ImageTraversal::Iterator::Iterator(ImageTraversal* it) {
  it_ = it;
  start_ = it_->getStart();
  curX = start_.x;
  curY = start_.y;
  png_ = it_->getPNG();
  tolerance_ = it_->getTolerance();
  width_ = png_.width();
  height_ = png_.height();
  for(unsigned i = 0; i < width_; i++){
    vector<bool> v;
    for(unsigned j = 0; j < height_; j++){
      v.push_back(false);
    }
    visited_.push_back(v);
  }
}

/**
 * Iterator increment opreator.
 *
 * Advances the traversal of the image.
 */
ImageTraversal::Iterator & ImageTraversal::Iterator::operator++(){
  Point temp;
  temp = it_->pop();
  curX = temp.x;
  curY = temp.y;
  visited_[curX][curY] = true;
  move();
  while(!it_->empty())
  {
    temp = it_->peek();
    if(!visited_[temp.x][temp.y]) break;
    it_->pop();
  }
  return *this;
}

/**
 * Iterator accessor opreator.
 *
 * Accesses the current Point in the ImageTraversal.
 */
Point ImageTraversal::Iterator::operator*() {
  return this->it_->peek();
}

/**
 * Iterator inequality operator.
 *
 * Determines if two iterators are not equal.
 */
bool ImageTraversal::Iterator::operator!=(const ImageTraversal::Iterator &other) {
  return !(this->it_->empty());
}

ImageTraversal::Iterator::~Iterator(){
  delete it_;
}

ImageTraversal::~ImageTraversal(){

}

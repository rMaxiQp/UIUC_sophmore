#include <iterator>
#include <cmath>
#include <list>
#include <stack>

#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"
#include "DFS.h"

/**
 * Initializes a depth-first ImageTraversal on a given `png` image,
 * starting at `start`, and with a given `tolerance`.
 */
DFS::DFS(const PNG & png, const Point & start, double tolerance) {
  png_ = png;
  start_ = start;
  tolerance_ = tolerance;
  s_.push(start);
}

/**
 * Returns an iterator for the traversal starting at the first point.
 */
ImageTraversal::Iterator DFS::begin() {
  DFS *dfs = new DFS(png_, start_, tolerance_);
  return ImageTraversal::Iterator(dfs);
}

/**
 * Returns an iterator for the traversal one past the end of the traversal.
 */
ImageTraversal::Iterator DFS::end() {
  return ImageTraversal::Iterator();
}

/**
 * Adds a Point for the traversal to visit at some point in the future.
 */
void DFS::add(const Point & point) {
  s_.push(point);
}

/**
 * Removes and returns the current Point in the traversal.
 */
Point DFS::pop() {
  Point temp = s_.top();
  s_.pop();
  return temp;
}

/**
 * Returns the current Point in the traversal.
 */
Point DFS::peek() const {
  return s_.top();
}

/**
 * Returns true if the traversal is empty.
 */
bool DFS::empty() const {
  return s_.empty();
}

//Getters
double DFS::getTolerance() const{
  return tolerance_;
}

Point DFS::getStart() const{
  return start_;
}

PNG DFS::getPNG() const{
  return png_;
}

DFS::~DFS(){}

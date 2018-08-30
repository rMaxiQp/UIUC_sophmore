
#include <iterator>
#include <cmath>
#include <list>
#include <queue>

#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"
#include "BFS.h"

using namespace cs225;
using namespace std;
/**
 * Initializes a breadth-first ImageTraversal on a given `png` image,
 * starting at `start`, and with a given `tolerance`.
 */
BFS::BFS(const PNG & png, const Point & start, double tolerance) {
  png_ = png;
  start_ = start;
  tolerance_ = tolerance;
  q_.push(start);
}

/**
 * Returns an iterator for the traversal starting at the first point.
 */
ImageTraversal::Iterator BFS::begin() {
  BFS* bfs = new BFS(png_, start_, tolerance_);
  return ImageTraversal::Iterator(bfs);
}

/**
 * Returns an iterator for the traversal one past the end of the traversal.
 */
ImageTraversal::Iterator BFS::end() {
  return ImageTraversal::Iterator();
}

/**
 * Adds a Point for the traversal to visit at some point in the future.
 */
void BFS::add(const Point & point) {
    q_.push(point);
}

/**
 * Removes and returns the current Point in the traversal.
 */
Point BFS::pop() {
  Point temp = q_.front();
  q_.pop();
  return temp;
}

/**
 * Returns the current Point in the traversal.
 */
Point BFS::peek() const {
  return q_.front();
}

/**
 * Returns true if the traversal is empty.
 */
bool BFS::empty() const {
  return q_.empty();
}

//Getters
double BFS::getTolerance() const{
  return tolerance_;
}

Point BFS::getStart() const{
  return start_;
}

PNG BFS::getPNG() const{
  return png_;
}


BFS::~BFS(){}

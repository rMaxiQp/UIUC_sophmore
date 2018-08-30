/**
 * @file kdtree.cpp
 * Implementation of KDTree class.
 */
 #include <cmath>
 using namespace std;

template <int Dim>
bool KDTree<Dim>::smallerDimVal(const Point<Dim>& first,
                                const Point<Dim>& second, int curDim) const
{
  if(first[curDim] == second[curDim]){//if first[curDim]  == second[curDim]
    return first < second;
  }
  return first[curDim] < second[curDim];
}

template <int Dim>
bool KDTree<Dim>::shouldReplace(const Point<Dim>& target,
                                const Point<Dim>& currentBest,
                                const Point<Dim>& potential) const
{
  double current = calculateDistance(target, currentBest);
  double possible = calculateDistance(target, potential);
  if(current == possible) return potential < currentBest; //when distance is same, compare by point
  return possible < current;
}

template <int Dim>
KDTree<Dim>::KDTree(const vector<Point<Dim>>& newPoints)//constructor
{
  points = newPoints;
  buildKD(0, points.size()-1, 0);
}

template <int Dim>
void KDTree<Dim>::buildKD(int left, int right, int dimension){
  if(left >= right) return;
  int median = (left + right)/2;
  select(left, right, median, dimension); //sort based on current point
  buildKD(left, median-1, (dimension+1)%Dim);  //sort left part
  buildKD(median+1, right, (dimension+1)%Dim); //sort right part
}

template <int Dim>
int KDTree<Dim>::partition(int left, int right, int median, int dimension)
{
  Point<Dim> midpoint = points[median];
  swap(points[median], points[right]); //move median to end
  int store = left;
  for(int current = left; current < right; current++)
  {
    if(smallerDimVal(points[current], midpoint, dimension)) //if current is smaller
    {
      swap(points[store], points[current]);
      store++;
    }
  }
  swap(points[right], points[store]); //restore median to the orignal place
  return store;
}

template <int Dim>
void KDTree<Dim>::select(int left, int right, int median, int dimension)
{
  /**base case*/
  if(left == right)
  {
    return;
  }
  /** recursive case*/
  int new_median = (left + right)/2;
  new_median = partition(left, right, new_median, dimension);

  if (median < new_median)  //left part
  {
    select(left, new_median-1, median, dimension);
  }
  else if(median > new_median) // right part
  {
    select(new_median+1, right, median, dimension);
  }
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim> & query) const
{
     Point<Dim> current;
     find(0, points.size()-1, 0, current, query); //find nearest neighbor helper function
     return current;
}


template <int Dim>
void KDTree<Dim>::find(int left, int right, int K, Point<Dim>& current, const Point<Dim>& target) const
{
  /** base case of the recursion*/
  if(left >= right)
  {
    if(left == 0)
    {
      current = points[left];
    }
    else if(shouldReplace(target, current, points[left]))
    {
      current = points[left];
    }
    return;
  }

  /** start of the recursive case*/
  int mid = (left + right) / 2;

  if(smallerDimVal(current, points[mid], K))
  {
    find(left, mid - 1, (K+1)%Dim, current, target);
    if(shouldReplace(target, current, points[mid])) //to the left branch since left is smaller
    {
      current = points[mid];
    }
    double radius = calculateDistance(target, current);
    if(inRange(target, points[mid], K, radius))
    {
      find(mid + 1, right, (K+1)%Dim, current, target);
    }
  }
  else //to the right branch since right is bigger
  {
    find(mid + 1, right, (K+1)%Dim, current, target);

    if(shouldReplace(target, current, points[mid]))
    {
      current = points[mid];
    }
    double radius = calculateDistance(target, current);

    if(inRange(target, points[mid], K, radius))//check lelft
    {
      find(left, mid - 1, (K+1)%Dim, current, target);
    }
  }
  return;
}

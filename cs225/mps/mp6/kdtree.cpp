/**
 * @file kdtree.cpp
 * Implementation of KDTree class.
 */

#include <utility>
#include <algorithm>

template <int Dim>
bool KDTree<Dim>::smallerDimVal(const Point<Dim>& first,
                                const Point<Dim>& second, int curDim) const
{
    /**
     * @todo Implement this function!
     */
    if (first[curDim] == second[curDim])
        return first < second;
    else
        return first[curDim] < second[curDim];
}

template <int Dim>
bool KDTree<Dim>::shouldReplace(const Point<Dim>& target,
                                const Point<Dim>& currentBest,
                                const Point<Dim>& potential) const
{
    /**
     * @todo Implement this function!
     */
    return (distanceSquared(target, potential) < distanceSquared(target, currentBest)
            || (distanceSquared(target, potential) == distanceSquared(target, currentBest)
                && potential < currentBest));
}

template <int Dim>
KDTree<Dim>::KDTree(const vector<Point<Dim>>& newPoints)
    : points(newPoints)
{
    /**
     * @todo Implement this function!
     */
    buildTree(0, 0, points.size() - 1);
}

template <int Dim>
void KDTree<Dim>::buildTree(int d, int left, int right)
{
    if (left < right) {
        // "Sort" the list in kD changing-dim space
        // by selecting the median to the median position
        // and then recursively sorting the two halves of the list

        int mid = (right + left) / 2;
        select(d, left, right, mid);

        buildTree((d + 1) % Dim, left, mid - 1);
        buildTree((d + 1) % Dim, mid + 1, right);
    }
}

template <int Dim>
bool compare(const Point<Dim>& first, const Point<Dim>& second, int curDim)
{
    return first[curDim] < second[curDim];
}

template <int Dim>
void KDTree<Dim>::select(int dim, int left, int right, int k)
{
    int pivotIndex;
    while (true) {
        pivotIndex = partition(dim, left, right, k);

        if (k == pivotIndex)
            return;
        else if (k < pivotIndex)
            right = pivotIndex - 1;
        else
            left = pivotIndex + 1;
    }
}


template <int Dim>
int KDTree<Dim>::partition(int dim, int left, int right, int pivotIndex)
{
    Point<Dim> pivotValue = points[pivotIndex];
    std::swap(points[pivotIndex], points[right]);

    //Point<Dim> c = points[pivotIndex];
    //points[pivotIndex] = points[right];
    //points[right] = c;

    int storeIndex = left;
    for (int i = left; i < right; i++) {
        if (smallerDimVal(points[i], pivotValue, dim)) {
            std::swap(points[storeIndex], points[i]);
            //Point<Dim> d = points[storeIndex];
            //points[storeIndex] = points[i];
            //d = points[i];
            ++storeIndex;
        }
    }

    std::swap(points[right], points[storeIndex]);
    //Point<Dim> e = points[right];
    //points[right] = points[storeIndex];
    //points[storeIndex] = e;
    return storeIndex;
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim>& query) const
{
    /**
     * @todo Implement this function!
     */
    int nearest = findNearest(query, 0, 0, points.size() - 1);
    return points[nearest];
}

template <int Dim>
int KDTree<Dim>::findNearest(const Point<Dim>& query, int dim, int left,
                             int right) const
{
    if (left >= right)
        return left;

    int mid = (right + left) / 2;
    int nearest, other;
    double radius2, split2; // '2' means 'squared distance'

    if (smallerDimVal(query, points[mid], dim)) {
        // Find the nearest neighbor in the left subtree.
        nearest = findNearest(query, (dim + 1) % Dim, left, mid - 1);
    } else {
        // Find the nearest neighbor in the right subtree.
        nearest = findNearest(query, (dim + 1) % Dim, mid + 1, right);
    }

    // Check if the current root is closer.
    if (shouldReplace(query, points[nearest], points[mid]))
        nearest = mid;

    // Check if we need to look in the other subtree.
    // We do this by seeing if the distance from the query and the current best
    // is larger than the distance from the query to the splitting plane. If it
    // is, then there is a chance that there is a point in the other subtree
    // that is closer than the current best.
    //
    // In 2-D, this might look like:
    //
    //   Need to search other subtree:              Don't need to search other subtree:
    //                       | splitting plane                                   | splitting plane
    //                       |                                                   |
    // current   . -- ~~~ -- .                    current   . -- ~~~ -- .        |
    // best  .-~             |@~-.                best  .-~               ~-.    |
    //    \ /                |@@@@\                  \ /                     \   |
    //     *                 |@@@@@\                  *                       \  |
    //    |          query   |@@@@@@|                |          query          | |
    //    |            *     |@@@@@@|                |            *            | |
    //    |                  |@@@@@@|                |                         | |
    //     \                 |@@@@@/                  \                       /  |
    //      \                |@@@@/                    \                     /   |
    //       `-.             |@.-'                      `-.               .-'    |
    //           ~- . ___ . -~                              ~- . ___ . -~        |
    //                       |                                                   |
    //                       |                                                   |
    //
    // If the '@' region exists (i.e., the circle intersects the splitting
    // line), then we have to search the other subtree, since there could be
    // another point (any of the '@'s) that is closer to the query than the
    // current best.
    //
    // In >2-d, the concept is the same, but instead of a circle intersecting a
    // line, it's a hypersphere intersecting a hyperplane.

    // First, compute the distance between the query and the current best.
    radius2 = distanceSquared(query, points[nearest]);

    // Now, compute the (perpendicular) distance between the query and the
    // splitting plane.
    // Instead of calculating this manually, we could also have done something
    // like:
    //    Point<Dim> perpendicularSplitPoint = query;
    //    query[dim] = points[mid][dim];
    //    split2 = distanceSquared(query, perpendicularSplitPoint);
    split2 = (points[mid][dim] - query[dim]);
    split2 *= split2;

    if (radius2 >= split2) {
        if (smallerDimVal(query, points[mid], dim)) {
            // We searched the left subtree before; now we need to search the
            // right subtree.
            other = findNearest(query, (dim + 1) % Dim, mid + 1, right);
        } else {
            // We searched the right subtree before; now we need to search the
            // left subtree.
            other = findNearest(query, (dim + 1) % Dim, left, mid - 1);
        }

        // We also need to check that the point we found in the other subtree is
        // actually better then our current best.
        if (shouldReplace(query, points[nearest], points[other]))
            nearest = other;
    }

    return nearest;
}

template <int Dim>
double KDTree<Dim>::distanceSquared(const Point<Dim>& first,
                                    const Point<Dim>& second) const
{
    double distsqr = 0;
    for (int i = 0; i < Dim; ++i)
        distsqr += (first[i] - second[i]) * (first[i] - second[i]);
    return distsqr;
}

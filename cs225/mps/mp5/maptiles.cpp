/**
 * @file maptiles.cpp
 * Code for the maptiles function.
 */

#include <iostream>
#include <map>
#include "maptiles.h"

using namespace std;

MosaicCanvas* mapTiles(SourceImage const& theSource,
                       vector<TileImage> const& theTiles)
{
/**initilization*/
 vector<Point<3>> v;
 map<Point<3>, TileImage> theMap; //Point<3> as key, TileImage as value
 unsigned row = theSource.getRows();
 unsigned column = theSource.getColumns();
 unsigned vertical = 0;
 unsigned horizontal = 0;

 for(unsigned long i = 0; i < theTiles.size(); i++)//push TileImages into vector by their pixel
 {
   HSLAPixel current = theTiles[i].getAverageColor();
   Point<3> p(current.h/360.0,  current.s, current.l);
   theMap[p] = theTiles[i];//link the key and value
   v.push_back(p);
 }

 KDTree<3> potential(v);//construct KDTree

 MosaicCanvas* field = new MosaicCanvas(row, column);
 //iteration to set tiles

 for(unsigned i = 0; i < row * column; i++)
 {
     vertical = i / row;
     horizontal = i % row;
     HSLAPixel current = theSource.getRegionColor(horizontal,vertical);
     Point<3> p(current.h/360.0,current.s, current.l);
     field->setTile(horizontal, vertical, theMap[potential.findNearestNeighbor(p)]);
 }
 return field;
}

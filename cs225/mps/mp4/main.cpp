
#include "cs225/PNG.h"
#include "FloodFilledImage.h"
#include "Animation.h"

#include "imageTraversal/DFS.h"
#include "imageTraversal/BFS.h"

#include "colorPicker/RainbowColorPicker.h"
#include "colorPicker/GradientColorPicker.h"
#include "colorPicker/GridColorPicker.h"
#include "colorPicker/SolidColorPicker.h"
#include "colorPicker/MyColorPicker.h"

using namespace cs225;

int main() {

  // @todo [Part 3]
  // - The code below assumes you have an Animation called `animation`
  // - The code provided below produces the `myFloodFill.png` file you must
  //   submit Part 3 of this assignment -- uncomment it when you're ready.

  PNG png;
  png.readFromFile("tests/i.png");

  FloodFilledImage image(png);
  DFS dfs(png, Point(40, 40), 0.05);
  BFS bfs(png, Point(170,200), 0.00001);
  MyColorPicker mcp;
  RainbowColorPicker rcp(0.1);
  image.addFloodFill( bfs, rcp );
  Animation animation = image.animate(1000);
  image.addFloodFill( dfs, mcp );
  Animation bnimation = image.animate(1000);
  unsigned i = 0;
  while(i < bnimation.frameCount()){
    animation.addFrame(bnimation.getFrame(i));
    i++;
  }
  PNG lastFrame = animation.getFrame( animation.frameCount() - 1 );
  lastFrame.writeToFile("myFloodFill.png");
  animation.write("myFloodFill.gif");

  return 0;
}

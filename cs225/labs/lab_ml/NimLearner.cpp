/**
 * @file NimLearner.cpp
 * CS 225 - Fall 2017
 */

#include "NimLearner.h"

using namespace std;

/**
 * Constructor to create a game of Nim with `startingTokens` starting tokens.
 *
 * This function creates a graph, `g_` representing all of the states of a
 * game of Nim with vertex labels "p#-X", where:
 * - # is the current player's turn; p1 for Player 1, p2 for Player2
 * - X is the tokens remaining at the start of a player's turn
 *
 * For example:
 *   "p1-4" is Player 1's turn with four (4) tokens remaining
 *   "p2-8" is Player 2's turn with eight (8) tokens remaining
 *
 * All legal moves between states are created as edges with initial weights
 * of 0.
 *
 * @param startingTokens The number of starting tokens in the game of Nim.
 */
NimLearner::NimLearner(unsigned startingTokens) : g_(true) {
  string p1 = "p1-";
  string p2 = "p2-";
  unsigned count = 0;
  for(; count <= startingTokens; count++)
  {
    g_.insertVertex(p1 + to_string(count));
    g_.insertVertex(p2 + to_string(count));
  }
  startingVertex_ = g_.getVertexByLabel("p1-" + to_string(count-1));

  Vertex player1;
  Vertex player2;

  while(count-- != 1)
  {
    player1 = g_.getVertexByLabel("p1-" + to_string(count));
    player2 = g_.getVertexByLabel("p2-" + to_string(count));
    if(count > 1)
    {
      g_.insertEdge(player2, g_.getVertexByLabel("p1-" + to_string(count - 2)));
      g_.setEdgeWeight(player2, g_.getVertexByLabel("p1-" + to_string(count - 2)), 0);
      g_.insertEdge(player1, g_.getVertexByLabel("p2-" + to_string(count - 2)));
      g_.setEdgeWeight(player1, g_.getVertexByLabel("p2-" + to_string(count - 2)), 0);
    }
    g_.insertEdge(player2, g_.getVertexByLabel("p1-" + to_string(count - 1)));
    g_.setEdgeWeight(player2, g_.getVertexByLabel("p1-" + to_string(count - 1)), 0);
    g_.insertEdge(player1, g_.getVertexByLabel("p2-" + to_string(count - 1)));
    g_.setEdgeWeight(player1, g_.getVertexByLabel("p2-" + to_string(count - 1)), 0);
  }
}

/**
 * Plays a random game of Nim, returning the path through the state graph
 * as a vector of `Edge` classes.  The `origin` of the first `Edge` must be
 * the vertex with the label "p1-#", where # is the number of starting
 * tokens.  (For example, in a 10 token game, result[0].origin must be the
 * vertex "p1-10".)
 *
 * @returns A random path through the state space graph.
 */
std::vector<Edge> NimLearner::playRandomGame() const {
  vector<Edge> path;
  Vertex current = startingVertex_;
  vector<Vertex> maybe = g_.getAdjacent(startingVertex_);
  while(maybe.size() > 1)
  {
    if(rand() % 2 == 1)
    {
      path.push_back(g_.getEdge(current, maybe[1]));
      current = maybe[1];
    }
    else
    {
      path.push_back(g_.getEdge(current, maybe[0]));
      current = maybe[0];
    }
    maybe = g_.getAdjacent(current);
  }
  if(maybe.size() == 1)
  {
    path.push_back(g_.getEdge(current, maybe[0]));
  }
  return path;
}


/*
 * Updates the edge weights on the graph based on a path through the state
 * tree.
 *
 * If the `path` has Player 1 winning (eg: the last vertex in the path goes
 * to Player 2 with no tokens remaining, or "p2-0", meaning that Player 1
 * took the last token), then all choices made by Player 1 (edges where
 * Player 1 is the source vertex) are rewarded by increasing the edge weight
 * by 1 and all choices made by Player 2 are punished by changing the edge
 * weight by -1.
 *
 * Likewise, if the `path` has Player 2 winning, Player 2 choices are
 * rewarded and Player 1 choices are punished.
 *
 * @param path A path through the a game of Nim to learn.
 */
void NimLearner::updateEdgeWeights(const std::vector<Edge> & path) {
  int p1_score = 1;
  int weight;
  string label = g_.getVertexLabel(path[path.size()-1].source);
  if(label[1] - '0' ==  2) //if player2 wins
  {
    p1_score = -1; //player1's weight turns to -1;
  }

  for(size_t t = 0; t < path.size(); t++){
    weight = g_.getEdgeWeight(path[t].source, path[t].dest); //get current weight
    label = g_.getVertexLabel(path[t].source);
    if(label[1] - '0' == 1) //player1
    {
      g_.setEdgeWeight(path[t].source, path[t].dest, weight + p1_score);
    }
    else //player2
    {
      g_.setEdgeWeight(path[t].source, path[t].dest, weight - p1_score);
    }
  }

  // back track
  // int i = 1;
  // for(int t = path.size() - 1; t >= 0; t--)
  // {
  //   weight = g_.getEdgeWeight(path[t].source, path[t].dest);
  //   g_.setEdgeWeight(path[t].source, path[t].dest, weight + i);
  //   i = -i;
  // }
}


/**
 * Returns a constant reference to the state space graph.
 *
 * @returns A constant reference to the state space graph.
 */
const Graph & NimLearner::getGraph() const {
  return g_;
}

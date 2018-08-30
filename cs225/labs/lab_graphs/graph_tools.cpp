/**
 * @file graph_tools.cpp
 * This is where you will implement several functions that operate on graphs.
 * Be sure to thoroughly read the comments above each function, as they give
 *  hints and instructions on how to solve the problems.
 */

#include "graph_tools.h"

/**
 * Finds the minimum edge weight in the Graph graph.
 * THIS FUNCTION IS GRADED.
 *
 * @param graph - the graph to search
 * @return the minimum weighted edge
 *
 * @todo Label the minimum edge as "MIN". It will appear blue when
 *  graph.savePNG() is called in minweight_test.
 *
 * @note You must do a traversal.
 * @note You may use the STL stack and queue.
 * @note You may assume the graph is connected.
 *
 * @hint Initially label vertices and edges as unvisited.
 */
int GraphTools::findMinWeight(Graph& graph)
{
  _initialize(graph);
  queue<Vertex> q;
  Edge min, temp;
  bool first = true;
  Vertex v;
  Vertex start = graph.getStartingVertex();
  vector<Vertex> v_list;

  graph.setVertexLabel(start, "VISITED");
  q.push(start);
  while(!q.empty())
  {
    v = q.front();
    q.pop();
    v_list = graph.getAdjacent(v);
    for(auto& w : v_list)
    {
      if(graph.getVertexLabel(w) == "UNEXPLORED")
      {
        graph.setEdgeLabel(v, w, "VISITED");
        graph.setVertexLabel(w, "VISITED");
        temp = graph.getEdge(v,w);
        if(first)
        {
          min = temp;
          first = false;
        }
        else if(temp < min)
        {
          min = temp;
        }
        q.push(w);
      }
      else if(graph.getEdgeLabel(v, w) == "UNEXPLORED")
      {
        graph.setEdgeLabel(v, w, "CROSS");
      }
    }
  }


  graph.setEdgeLabel(min.source, min.dest, "MIN");

  return min.weight;
}

/**
 * Returns the shortest distance (in edges) between the Vertices
 *  start and end.
 * THIS FUNCTION IS GRADED.
 *
 * @param graph - the graph to search
 * @param start - the vertex to start the search from
 * @param end - the vertex to find a path to
 * @return the minimum number of edges between start and end
 *
 * @todo Label each edge "MINPATH" if it is part of the minimum path
 *
 * @note Remember this is the shortest path in terms of edges,
 *  not edge weights.
 * @note Again, you may use the STL stack and queue.
 * @note You may also use the STL's unordered_map, but it is possible
 *  to solve this problem without it.
 *
 * @hint In order to draw (and correctly count) the edges between two
 *  vertices, you'll have to remember each vertex's parent somehow.
 */
int GraphTools::findShortestPath(Graph& graph, Vertex start, Vertex end)
{
  _initialize(graph);
  queue<Vertex> q;
  unordered_map<Vertex, Vertex> um;
  Vertex v;
  vector<Vertex> v_list;

  graph.setVertexLabel(start, "VISITED");
  q.push(start);
  while(!q.empty())
  {
    v = q.front();
    q.pop();
    v_list = graph.getAdjacent(v);
    for(auto& w : v_list)
    {
      if(graph.getVertexLabel(w) == "UNEXPLORED")
      {
        graph.setEdgeLabel(v, w, "VISITED");
        graph.setVertexLabel(w, "VISITED");
        um.insert({w,v});
        q.push(w);
      }
      else if(graph.getEdgeLabel(v, w) == "UNEXPLORED")
      {
        graph.setEdgeLabel(v, w, "CROSS");
      }
    }
  }

  int min = 0;
  while(start != end)
  {
    graph.setEdgeLabel(end, um[end], "MINPATH");
    end = um[end];
    min++;
  }
  return min;
}

/**
 * Finds a minimal spanning tree on a graph.
 * THIS FUNCTION IS GRADED.
 *
 * @param graph - the graph to find the MST of
 *
 * @todo Label the edges of a minimal spanning tree as "MST"
 *  in the graph. They will appear blue when graph.savePNG() is called.
 *
 * @note Use your disjoint sets class from MP 7.1 to help you with
 *  Kruskal's algorithm. Copy the files into the libdsets folder.
 * @note You may call std::sort instead of creating a priority queue.
 */
void GraphTools::findMST(Graph& graph)
{
  vector<Edge> edges = graph.getEdges();
  vector<Vertex> vertices = graph.getVertices();
  std::sort(edges.begin(), edges.end());
  DisjointSets ds;
  ds.addelements(vertices.size());
  for(auto e : edges)
  {
    Vertex u = e.source;
    Vertex v = e.dest;
    if(ds.find(u) != ds.find(v))
    {
      graph.setEdgeLabel(u, v, "MST");
      ds.setunion(u,v);
    }
  }
}

 /**
  * initialize all edges and vertices to unexplored
  * @param graph the graph to be initialized
  */
void GraphTools::_initialize(Graph& graph)
{
 vector<Edge> edges = graph.getEdges();
 for(Edge e : edges)
 {
   graph.setEdgeLabel(e.dest, e.source, "UNEXPLORED");
 }
 vector<Vertex> vertices = graph.getVertices();
 for(Vertex v : vertices)
 {
   graph.setVertexLabel(v, "UNEXPLORED");
 }
}

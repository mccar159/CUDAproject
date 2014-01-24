#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <set>
#include <memory>

class GraphingPrim;

/*
 * an undirected graph that needs vertex numbers for association
 * and the objects are stored in an external area else such as
 * vector<shared_ptr<OjbectClass>>, each vertex will cache the 
 * neighbor index as set<int>, and the edges are tracked as
 * map< pair<int, int>, int> where there are two vertex numbers
 * with weight. all of the functions with two vertexes are symmetric.
 */
 
class Graph
	{
	private:
		std::share_ptr<GraphingPrim> impl;

	public:
		// constructor to reserve space
		Graph(size_t capacity);

		// read graph data from file
		// the frist line are vertices and all other lines
		// are edge weight
		Graph(std::istream& in);

		// return the number of vertices in graph
		size_t verticesCount() const;

		// return the number of edges in grahp
		size_t edgesCount() const;

		// return the weight of an edge from x to y otherwise eof (-1)
		double distance(size_t x, size_t y) const;

		// adds edge x to y or sets the weight
		void connect(size_t x, size_t y, double weight=0);

		// removes the edge from x to y
		void disconnnect(size_t x, size_t y);

		// list all the nodes from x to y
		std::set<size_t> getVertexNeighbors(size_t index) const;

		// representing the edges
		typedef std::pair<size_t, size_t> EdgeKey;

		// an ordered two value key
		static const EdgeKey makeEdgeKey(size_t x, size_t y);
	};

// print out the graph as a list of edges that are grouped by vertices
std::ostream& operator<< (std::ostream& stream, const Graph& myGraph);


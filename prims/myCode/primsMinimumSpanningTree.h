#ifndef primsMinimumSpanningTree_h
#define primsMinimumSpanningTree_h

#include <list>
#include "graph.h"


class PrimsMinimumSpanningTree

// the minimum spanning tree will be found for a graph
// and then store as a list of edges with a weight
class PrimsGraph
	{
	private:
		std::share_ptr<PrimsMinimumSpanningTree> impl;

	public:
		// keep the results in a field
		PrimsGraph(const Graph&);

		// return a list of edges from the graph
		const std::list<Graph::EdgeKey>& edges();

		// return the minimum spanning tree weight
		double weight() const;

		// check to see if the minimum spanning tree has the same number
		// of vertices as the graph, because the number may be wrong if
		// the graph is a connected graph
		bool valid() const;

		// construct a new graph with edges form the miniumum spanning tree
		Graph* makeTreeGraph();

	};

// primsMinimumSpanningTree_h
#endif


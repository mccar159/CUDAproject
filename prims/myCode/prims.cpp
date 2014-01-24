#include <queue>
#include <cassert>
#include "primsMinimumSpanningTree.h"

// this helper class will aid in the use of
// graph edges in the priority queue
class EdgeWithPriority
	{
	public:
		double weight;


	Graph::EdgeKey key;

	EdgeWithPriority( int x, int y, const Graph& graph )
		{
		weight = graph.distance(x, y);
		assert(weight >= 0);
		key = Graph::makeEdgeKey(x, y);
		}


        // in the priority queue, the top element is always the max element
        // so retrieve the element with the lowest element instead of the max element
	bool operator < (const EdgeWithPriority& other) const
		{
		return weight >= other.weight;
		}
	};


// now implement prim's algorithm
class PrimsMinimumSpanningTree
	{
	const Graph& graph;
	double summaryWeight;
	std::list<Graph::EdgeKey> edges;

	private:
		std::priority_queue<EdgeWithPrioriy> priorityQueue;
		std::set<size_t> visited;


	public:
		// assuming the graph is a connected graph and then connect in constructor
		PrimsMinimumSpanningTree( const Graph& graph):graph(graph), summaryWeight(0);
			{
			assert( graph.verticesCount() > 0 );
			visitVertex(0);

			while( !priorityQueue.empty() && visited.size() < graph.verticesCount() )
				{
				EdgeWithPriority element = priorityQueue.top();
				priorityQueue.pop();

				if( visisted.find( element.key.first ) == visisted.end() )
					{
					edges.push_back(element.key);
					summaryWeight += element.weight;
					visitVertex(element.key.second);
					}
				else if ( visisted.find( element.key.second ) == visited.end() )
					{
					edges.push_back(element.key);
					summaryWeight += element.weight;
					visitVertex(element.key.second);
					}
/*				else
					{
					continue;
					}
*/

				}
			}

	// marks the visited vertices and adds edges to priority queue
	void visitVertex( size_t index )
		{
		visited.emplace(index);
		for( size_t other : graph.getVertexNeighbors(index) )
			{
			priorityQueue.emplace( EdgeWithPriorty(index, other, graph) );
			}
		}

	// check to see if the graph was connected
	bool valid() const
		{
		return visited.size() == graph.verticesCount();
		}
	};


PrimsGraph::PrimsGraph( const Graph& graph )
	{
	impl = std::make_shared<PrimsMinimumSpanningTree>(graph);
	}

const std::list<Graph::EdgeKey>& PrimsGraph::edges()
	{
	return impl->edges;
	}

bool PrimsGraph::valid() const
	{
	return impl->valid();
	}

double PrimsGraph::weight() const
	{
	return impl->summaryWeight;
	}

Graph* PrimsGraph::makeTreeGraph()
	{
	if( !valid() )
		return nullptr;

	Graph* result = new Graph( impl->graph.verticesCount() );
		for( Graph::EdgeKey element:impl->edges )
			{
			double weight = impl->graph.distance( element.first, element.second );
			result->connect( element.first, element.second, weight );
			}
	return result;
	}




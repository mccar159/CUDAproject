#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <cassert>

#include "graphForPrim.h"

// 2 value key that's ordered
const Graph::EdgeKey Graph::makeEdgeKey( size_t x, size_t y )
	{
	return (x > y) ? std::make_pair(y, x) : std::make_pair(x, y);
	}

/*
 * an undirected graph that needs vertex numbers for association
 * and the objects are stored in an external area else such as
 * vector<shared_ptr<OjbectClass>>, each vertex will cache the 
 * neighbor index as set<int>, and the edges are tracked as
 * map< pair<int, int>, int> where there are two vertex numbers
 * with weight. all of the functions with two vertexes are symmetric.
 *
 */
class GraphingPrim
	{
	private:
		size_t size;
		std::map<const Graph::EdgeKey, double> edges;
		std::vector<std::set<size_t>> vertexNeighbors;

	public:
		// reserve space for the capacity of the vertexes
		GraphingPrim(size_t capacity) : size(capacity), vertexNeighbors(capacity, std::set<size_t>() )
			{
			}

		// return the number of vertices in the graph
		int verticesCount() const
			{
			return size;
			}

		// return the number of edges in the graph
		int edgesCount() const
			{
			return edges.size();
			}

		// return the weight of the edge from node x to node y or eof (-1)
		double distance(size_t x, size_t y) const
			{
			// return false if there are no idices
			assert( x<size && y<size );
			auto nthElement = edges.find( Graph::makeEdgeKey(x, y) );
			if ( nthElement == edges.end() )
				{
				return -1;
				}
			else
				{
				return nthElement->second;
				}
			}

		// add the edge from x to y to the graph or set the new existing edge weight
		void connect( size_t x, size_t y, double weight = 0 )
			{
			assert( x<size && y<size )
			edges.insert( std::make_pair( Graph::makeEdgeKey(x, y), weight) );
			vertexNeighbors[x].insert(y);
			vertexNeighbors[y].insert(x);
			}

		// remove any of the edges from x to y
		void disconnect( size_t x, size_t y )
			{
			assert( x<size && y<size );
			edges.erase( Graph::makeEdgeKey(x, y);
                        vertexNeighbors[x].erase(y);
                        vertexNeighbors[y].erase(x);
			}

		// list all the nodes for edges from x to y
		std::set<size_t> getVertexNeighbors(size_t index) const
			{
			return vertexNeighbors.at(index);
			}

	};


std::ostream& operator<< ( std::ostream& stream, const Graph& myGraph )
	{
	size_t V = myGraph.verticesCount();
	size_t E = myGraph.edgesCount();

	stream << "This graph has " << V << " vertices and " << E << " edges." << std::endl;

	for( size_t i=0; i<V; i++ )
		{
		stream << " The edges " << i << ": ";
		for( auto priority : myGraph.getVertexNeighbors(i) )
			{
			stream << priority << "[ " << myGraph.distance(i, priority) << " ]";
			}
		stream << std::endl;
		}
	return stream;
	}


// make the graph
Graph::Graph( size_t capacity )
	{
	impl - std::make_shared<GraphingPrim>(capacity);
	}


// check to see if there is a connection and if there is a 
// connection, then connect x to y and assign a weigh
Graph::Graph( std::istream& in )
	{
	assert(in);
	size_t capacity;
	in >> capacity;
	impl = std::make_share<GraphingPrim>(capacity);

	while(in)
		{
		size_t x,y;
		double weight;
		in >> x;
		if(!in)
			break;
		in >> y;
		if(!in)
			break;
		in >> weight;
		impl->connect( x, y, weight );
		}
	}


size_t Graph::verticesCoutn() const
	{
	return impl->verticesCount();
	}

size_t Graph::edgesCount() const
	{
	return impl->edgesCount();
	}

double Graph::distance( size_t x, size_t y) const
	{
	return impl->distance(x, y);
	}

void Graph::connect( size_t x, size_t y, double weight )
	{
	impl->connect( size_t x, size_t y );
	}

void Graph::disconnect( size_t x. size_t y )
	{
	impl->disconnect( size_t x, size_t y );
	}

std::set<size_t> Graph::getVertexNeighbors(size_t index) const
	{
	return impl->getVertexNeighbox(index);
	}

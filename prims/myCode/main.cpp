#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "graph.h"
#include "primsMinimumSpanningTree.h"

using namespace std;

void basic_use()
// This function gives a simple and straightforward example
// of how graph classes API should work.
// In real code  graph vertices are associated with some complex data structures
// from some external storage. I recomend smth. like vector <shared_ptr<ObjectType>>.
{
    const int VERTICE_COUNT = 3;

    Graph G(VERTICE_COUNT);

    for(int i = 1; i < VERTICE_COUNT; i++)
    {
        G.connect(0, i, i);
    }
    ::cout << G;

    ::cout << DijkstraGraphPath(G, 0, 1);
}

void simulation(int node_count, double edge_density, int weight_limit)
// a graph with node_count vertices each of which have edge_density * (node_count - 1) neighbors
// each graph eadge have its weight between 1 and weight_limit
// when graph is created we search for the multiple paths:
// 0 to 1, 0 to 2, 0 to 3, 0 to 4, …, 0 to (node_count-1) and compute average path weight value
{
    ::cout << "Simulation for "
           << node_count << " nodes "
           << edge_density << " dencity and "
           << weight_limit << " max edge weight" << ::endl << ::endl;

    Graph G(node_count);

    ::srand(::time(NULL));

    for(int i = 0; i < node_count; i++)
        for(int j = i+1; j < node_count; j++)
        {
            double roll = static_cast<double>(::rand()) / RAND_MAX;
            if(roll < edge_density)
            {
                double weight = ::rand() % weight_limit + 1;
                G.connect(i, j, weight);
            }
        }

    ::cout << G << ::endl;
    ::cout << "Finding paths..." << ::endl << ::endl;

    double paths_summ = 0;
    int paths_count = node_count - 1;
    for(int i = 1; i < node_count; i++)
    {
        DijkstraGraphPath path(G, 0, i);
        if(path.exists())
            paths_summ += path.weight();
        else
            paths_count--;

        ::cout << path;
    }
    ::cout << "Average path length is " << paths_summ / paths_count << ::endl;
}

void something_wrong()
// replace standard terminate handler to set breakpoint here
// for debugging purpose
{
    ::cout << "bang!" << ::endl;
    ::abort();
}

int main()
{
    // programm output is massive, redirect to file for better experience
    ::set_terminate(something_wrong);

    /*  it was Assignment 2
    ::cout << "Part 1" << ::endl;
    simulation(50, 0.2, 10);

    ::cout << ::endl << "Part 2" << ::endl;
    simulation(50, 0.4, 10);
    */

    //const char SAMPLE_GRAPH_FILE[] = "SampleTestData.txt";
    //const char SAMPLE_GRAPH_FILE[] = "TinyDataDouble.txt";
    const char SAMPLE_GRAPH_FILE[] = "10000EWG.txt";

    ::cout << "Constructing graph object from " << SAMPLE_GRAPH_FILE << ::endl;
    ::filebuf fb;
    if (fb.open (SAMPLE_GRAPH_FILE,::ios::in))
    {
        std::istream in(&fb);
        Graph G(in);

        ::cout << G << ::endl << ::endl;

        PrimGraphMst mst(G);

        ::cout << "MST is "
               << (mst.valid() ? "valid" : "not valid")
               << " weight = " << mst.weight() << ::endl;

        ::shared_ptr<Graph> mst_graph(mst.makeTreeGraph());

        ::cout << *mst_graph;
    }
    else
    {
        ::cout << "Can't open file"  << ::endl;
    }

    return 0;
}





#include <iostream>

class primsAlgorithm
	{
	private:
		int numberOfNodes;
		int edgesInGraph[250][4];
		int numberOfEdgesInGraph;
		int edgesInTree[250][4];
		int numberOfEdgesInTree;
		int initalNode;

	// partition the graph into two graph sets
	// set 1 is called set1
	// set 2 is called set2
	int edgeOnTree1[50], treeEdges1;
	int edgeOnTree2[50], treeEdges2;

	public:
		void input();
		int findSet(int);
		void algorithm();
		void outupt();
	};

	void primsAlgorithm::input();
		{
		cout << "This output compares the Graphics Processing Unit (GPU)\n
		vs the Central Processing Unit (CPU).\n" << endl;

		cin >> numberOfNodes;

		numberOfEdgesInGraph = 0;

		cout << "Enter the weight for following edges in the graph ::\n" << endl;
		for( int i=1; i<=numberOfEdgesInGraph; i++ )
			{
			for( int j=1; j<=numberOfEdgesInGraph; j++)
				{
				cout << "<" <<i<< "," <<j<< "> ::";
				int w;
				cin >> w;
				if( w != 0 )
					{
					numberOfEdgesInGraph++;

					edgesInGraph[numberOfEdgesInGraph][1]=i;
                                        edgesInGraph[numberOfEdgesInGraph][2]=j;
                                        edgesInGraph[numberOfEdgesInGraph][3]=w;
					}
				}
			}
		// print the graph edges
		cout << "The edges of the graph are ::\n";
		for( i=1; i<=numberOfEdgesInGraph; i++)
			{
			cout << "<" <<edgesInGraph[i][1] 
                        << ", " <<edgesInGraph[i][2];
                        << "> ::" <<edgesInGraph[i][3] << endl;
			}

		int primsAlgorithm::findSet(int x)
			{
			
			}


		}

# Lab3 - AI Search Techniques
## Graph Model
 - Graph model
 - G = (V,E)
 - V is a set of vertices(nodes)
 - E is a set of edges; each edge is a pair of vertices
 - Example:
   - V = {1,2,3,4}
   - E = {(1,2),(1,3),(2,4),(4,1)}

![](/Lab3/Picture1.png)

### Undirect graph
 - The edges do not have a specific direction
   - Means edges can travel in both ways
   - E.g. Facebook friend relationship

### Directed graph
 - Edges have a specific direction
   - each edge can travel in one way
   - E.g. Instagram follower relationship

## Adjacency Matrix
 - The data structure to express a graph
 - It is a [V] x [V] matrix
 - Example:
![](/Lab3/Picture2.png)

## Adjacency List
 - a list ata structure to represent a graph
 - Vertex u has its adjacency list Ajd[u], which contains each vertex v being adjacent to u
 - E.g., Adj[4] = {1,2}
![](/Lab3/Picture3.png)

## Graph Analytics
 - Searching
 - Clustering/Partitioning
 - Shortest path
   - The shortest path between two vertices
 - Network analysis
 - etc...

## Uniformed Search & Informed Search
 - Uniformed Search:
   - also known as blind search
   - don't use any domain-specific knowledge or information to uide the search
     - Is often used when little or no information is acailable about the problem domain.
     - E.g., Breath-first search, Dijkstra's Algorithm

 - Informed Search:
   - Known as heuristic search
   - uses domain-specific knowledge or information(usuallt heuristic functions) to guide the search
     - E.g., A* search, greedy search

## Breadth-first Search
**Problem:**
 - Given a source vertex s in a graph G = (V,E)
 - Visit all vertices closer to s before any father vertices
 - Idea:
   - Explore the vertices of a graph according to "layers"
   - Layer 0 consists only of the starting vertex s
   - Layer 1 contains the neighbor vertices of s
   - Layer2 comprises the neighbors of layer-1
   - vertices that do not already belong to layer 0 or 1
   - Layer 3, so on and so forth
![](/Lab3/Picture4.jpg)

**Algorithm**
 - Mark all nodes as not visited
 - Mark startNode visited
 - Add startNode to the queue
 - While(queue is not empty):
   - Remove currentNode from queue
   - for each nextNode connected to currentNode:
     - If nextNode is not visited
     - Mark nextNode visited
     - If nextNode is goalNode, terminate with success
     - Add nextNode to queue
 - After all nodes in queue checked, terminate with failure
 - If terminate with success, return the path found

**Successful BFS**
![](/Lab3/Picture5.png)
 - Example:Search from B to G
   - Visited nodes {B}, queue {B}
 - Removed B from queue, insert all connected nodes
   - Visited nodes {BC}, queue{C}
 - Remove C from queue, insert all connected nodes
   - Visites {BCADE}, queue {ADE}
 - Remove A from queue, insert node H (not C, as it had meen visited)
   - Visited {BCADEH}, queue {DEH}
 - Remove D from queue, insert F
   - Visited {BCADEHF}, queue {EHF}
 - Remove E, add nothing(no othwe nodes to go)
   - Visited nodes {BCADEHF}, queue {HF}
 - Remove H, add G and the goal is found
   - Visited {BCADEHFG}, queue {FG}
 - **PATH: B > C > D > F > G**

**Unsuccessful BFS**
![](/Lab3/Picture6.png)
 - Search from B to J
 - We start from the last interation in the last example
   - Visited nodes {BCADEHFG}, queue {FG}
 - Remove F from queue, no additional node added
   - Visited nodes {BCADEHFG}, queue {G}
 - Remoe G from queue, no additional node added
   - Visited nodes {BCADEHFG}, queue {}
 - The queue is now empty
 - Therefore, no path exist from B to J
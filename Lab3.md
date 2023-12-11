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
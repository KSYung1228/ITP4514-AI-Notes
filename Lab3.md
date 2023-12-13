# Lab3 - AI Search Techniques
## Graph Model
 - Graph model
 - G = (V,E)
 - V is a set of vertices(nodes)
 - E is a set of edges; each edge is a pair of vertices
 - Example:
   - V = {1,2,3,4}
   - E = {(1,2),(1,3),(2,4),(4,1)}

![](/Lab3/Picture1.png){width=300}

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
---
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

## Advantages and Disadvantages of BFS
 - Advantages
   - Simple algorithm
 - Disadvantages
   - Not efficient
   - requires a lot of memory space
   - Not always getting the best path
---
## Dijkstra's algorithm and Uniform cost Search
 - finds the shortest path by picking the unvisited node with the lowest distance/ cost from start, calculates the distance through it to each unvisited neighbour, and updates the neighbour's distance if smaller

**Algorithm**
![](/Lab3/Picture7.png)
 - Suppose we have the following undirected graph. the initial node is A; the goal node is E
 - Create table to see the shortest distance for each traversal.
 - Traverse from node A, the path cost from node A > A = 0; alos assume the initial path cost to all other nodes is a very large value, say, infinity
   - Source vertex: A
![](/Lab3/Picture8.png)
 - From graph, A can traverse to B, C, D.
 - update the table with the shorest distance.
   - Mark A as visited
   - Souce Vertex: A
   - Run the algorithm for finding the shortest distance:
     - IF:
     $$
     d[i] + c(i, j) < d(j)
     $$
     - THEN:
     $$
     d(j) = d(i) + c(i, j)
     $$
![](/Lab3/Picture9.png)
 - Since A > C is 2, which is the shortest distance when compared with B and D.
 - Traverse from A to C
 - Mark C is visited
 - Next step, C can traverse to B, D
   - Run the algorithm for finding the shortest distance:
     - IF:
     $$
     d[i] + c(i, j) < d(j)
     $$
     - THEN:
     $$
     d(j) = d(i) + c(i, j)
     $$
![](/Lab3/Picture10.png)
 - Traverse from C to D
 - Update te table with the shortest path cost
![](/Lab3/Picture11.png)
 - Since the path cost A > C > B would be cheapter than A > C > D > B
   - Backward to C
   - Traverse to B
![](/Lab3/Picture12.png)
 - Update the path cost from B > E
 - Finally shortest path found
 - A > C > B > E, with path cost = 6
![](/Lab3/Picture13.png)

## Uniform Cost Search
**Uniform cost search algorithm**
 - Make use of priority queue
 - Starting from starting node, add all adjacent nodes to the priority queue, remove one with least cost
 - repeat dor the node with the least cost
 - When the goal is reached, continue (as there may be alternative paths with lower costs)
 - stop when the goal node is removed
 - Obtain the optimal path

## Advantages and Disadvantages of Uniform Cost Search
 - Advantages
   - Optimal path obtained(with minimum cost)
 - Disadvantages
   - Not efficient, often need more steps to find a path
   - May stick to an infinity loop if there is a path with infinite zero cost sequence
---
## Best-first Search
 - **Greedy algorithm**
   - Expanding the most promising node chosen according to a specified rule
 - In Best first search, node selection on every step is based on minimizing a heuristic function h(n)
 - The heuristic function h(n) is a cost estimate. E.g., it may represent the physical distance/ time/ expenditure required, or the energy consumed when a monster in a game move feom the node to the goal
 - Uses distance to goal not distance from start
 - Best first search uses the estimated distance to goal instead of the distance from the starting point in searching
 - Obstacles can cause a best first search to double vack, It does not always return the best path
 - In games, it is suitable for hand-held game with limited processing power

**Algorithm**
![](/Lab3/Picture14.png)
 - Find a good path from S to G
 - Given the follow heuristic cost information:
![](/Lab3/Picture15.png)
 - Make use of priority queue
 - Starting from starting node, add all adjacent nodes to the priority queue, calculate the distance from goal, remoce one with least value for the node just removed
 - Stop when the goal is reached
 - Obtain the path

```markdown
# Steps
Expand the nodes of S and put in the CLOSED list

Initialization: Open {A, b}, Closed {s}

Iteration 1: Open {A}, Closed {S, B}

Iteration 2: Open {E, F, A} Close {S, B}
           : Open {E, A} Close {S, B, F}

Iteration 3: Open {I, G, E, A}, Close {S, B, F}
           : Open {I, E, A}, Close {S, B, F, G}

Hence the final sulution path will be: S > B > F > G
```

## Advantages and Disadvantages of Best First Search
 - Advantages
   - Very efficient, fast exploration of node to reach the goal
   - Having low memory requirements
 - Disadvantages
   - Does not guarantee the best path, but often return a best path under certain conditions, e.g., in games with convex obstacles only
   - Likely to have inaccrate results
---
## A* Search
 - Combination of Dijkstra's algorithm and best first search
 - The first path is the shortest path
 - A* uses distance heuristic to push search towards goal
 - Total cost from start to goal = actual shortest distance from start to current + estimated(heuristic) distance from current node to goal
 - In short, f(start, goal) = g(start, curent) + h(current, goal)
 - h(current, goal) = cost of the cheapest path from current node to the goal node
 - g(start, current) = cost of the cheapest path from the starting node to current node

**Algorithm**
 - Make use of priority queue
 - Starting from startinf node, add all adjacent nodes to the priority queue, calculate the value of the "heuristic function" (e.g., distance from goal + cost) remove one with least value for the heuritic function
 - repeat dor the node just removed
 - When the goal is reached, continue(as there may be alternative paths with lower costs)
 - Stop when the goal node is removed

![](/Lab3/Picture16.png)

![](/Lab3/Picture17.png)

 - In calculation(from S > G)
  - f(S, A) = 3 + 10.2 = 13.2
  - f(S, D) = 4.8 + 9 = 13.8
  - f(S, B) = 3 + 4 + 6 = 13
  - f(S, D) = 3 + 5 + 9 = 17
  - f(S, C) = 3 + 4 + 4 + 3 = 14
  - f(S, E) = 3 + 4 + 5 + 7 = 19
  - f(S, A) = 4 + 5 + 10 = 19
  - f(S, E) = 4 + 2 + 7 = 13
  - f(S, B) = 4 + 2 + 5 + 6 = 17
  - f(S, F) = 4 + 2 + 4 + 4 = 14
  - f(S, G) = 4 + 2 + 4 + 4 + 0 = 14
  - The total cost from the start node S to the goal node G is 14

**Explane A\* Search**
```py
#define the grid
grid = [[0,1,0,0,0,0],
        [0,1,0,0,0,0],
        [0,1,0,0,0,0]
        [0,1,0,0,0,0],
        [0,0,0,0,1,0]]
'''
5*6 maze
1: cannot go through
0: can go through
'''
#define the heuristics and parameters
init = [0,0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 #Same as Best First Search execpt: cost = 0 > cost = 1
delta = [[-1,0], # go up
         [0,-1], # go left
         [1,0],  # go down
         [0,1]]  # go right

delta_name = ['^','<','v','>']
#define the algorithm
if x == goal[0] and y == goal[1]:
    found = True
else:
    for i in tange(len(delta)):
        x2 = x + delta[i][0]
        y2 = y + delta[i][1]
        if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
            if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                g2 = g + cost # same as Best First search expert: g2 = g
                f = g2 + heuristic[x2][y2]
                open.append([f, g2, x2, y2])
                closed[x2][y2] = 1
return expand


result = search(grid, init, goal, cost, heuristic)

for el in result:
    print(el)
```
##Comparisons on Best-first search and A-Star Search
 - Similarities:
   - Both best-first search and A* search use an exploratuib strategy that selects the most promising node based on a heuristic ecaluation
   - Both algorithms maintain a priority queue to store the nodes yet to be expanded.
 - Differences:A* search guarantees finding an optimal solution if the heursir function is afmissible. Best-first search does not guarantee optimality unless additiona conditions are met.
 - Best-first search only considers the heuristic value of each node when selecting the next node to expand;A* Search takes into account both the heurustic value and the cost of reaching each node.
 - A* search is complete, meaning it will find a solution if one exists. Best-first search does not gunarantee completeness.
![](/Lab3/Picture18.png)
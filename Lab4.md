# Lab4 - Logic Programming & CSP
## Constraint Satisfaction Problem (CSP)
 - Class of problems in computer science and artificial intelligence that incolbe finding solutions that satisfy a set of constraints
 - A problem is solved when each variable has a value that satisfies all the constraints on the variable
 - The main idea is to eliminate large portions of the search space all at once by **identifying variable/ value combinations that voilate the constraints**

## CSP components
 - Variables
 - Domains
 - Constraints

**Variables**
 - represents an entity or a quantity that we want to assign a value to in order to find a solution that satisfies the given constraints
 - Can take different orms depending on the nature of the problem
   - They can represent discrete quantities, such as numbers, colorsm or symbols, or they can represent cuntinuous quantities with specified ranges or intervals
 - E.g., ***each cell in sudoku*** represents a variable.

**Domians**
 - Each variable in a CSP has an associated domain, which is a set of possible values that the variable can take
 - In sudoku puzzle, the ***domain*** of each variable is ***the set of numbers from 1 to 9***

**Constraint**
 - Constraint define rules and conditions that the possible cumbinations of values for the variables
 - In sudoku, some **<u>example constraints**</u> are:
   - Row constraint: each row in sudoku must bontain all digits from 1 to 9; none of these digits could be repeated
   - Column constraint
   - Box/Squarec onstraint: Each 3x3 bix in sudoku grid must contain all the digits from 1 to 9 exactly once; none of these digits could be repeated
   
**Constraint: Some variations**
   - Constraints can take various forms depending on the nature of the problem
   - Some common types of constaints include:
     - **Unary constraints** - incolces a single variable only (e.g., the value of "X" must been even)
     - **Binary Constraints** - the restrictions incolce two vaiables (e.g., "X" and "Y" must be different)
     - **Global Constraints** - involves multiple variables (e.g., "all-different constraint" - as its name implies, all the variables in the problem must be different)

## Logic Programming
 - Logic programming is a programming paradigm that is based on formal logic and uses logical rules and inference to solve problems
 - In logic programming, progra,s are composed of facts and rules expressed in a logic-based language
 - Th most common logic programming language is  ***Prolog***, which stands for "Programming in Logic" In Prolog, you define a knowledge base consisting of facts and rules, and you can query this knowledge base to obtain solutions to problems.

**Logic Programming: An Overview**
 - Logic Programming uses ***facts*** and ***rules*** for solving the problem
 - That is why they are called the ***building blocks*** of logic programming
 - A ***goal*** needs to be specified for every program in logic programming

**Facts**
 - basically are true statements ***about the program and data***
 - For example, Beijing is the capital of china

**Rules**
 - Rules are the constraints which allow us to make conclusions about the problem domain.
 - Rules basically written as logical clauses to express various facts
 - E.g., If we building any game then all the rules must be defined
 - Important to solve ant problem in Logic Programming. Rules are basically
 - logical conclusion which can express the facts, Following is the syntax of rule:
 
 $$
 A:  B_1,B_2,...,B_n
 $$
 
 - Here, A is the head and B<sub>1</sub>,B<sub>2</sub>,...B<sub>n</sub> is the body
 - E.g.,

$$
ancenstor(X,Y): father(X,Y)
$$

$$
ancenstor(X,Z): father(X,Y), ancestor(Y,Z)
$$

**Logic Program in Prolog:**

```py
#facts(database)
parent(joe, jane)
parent(harry, carl)
parent(meg, jane)
parent(jane, anne)
parent(carl, ralph)
parent(hazel, harry)

#Rule
grandparent(X, Z):
    parent(X,Y),parent(Y,Z)

#Queries
? parent(meg, jane)
#True

? parent(carl,joe)
#False

? granfparent(A, ralph)
A = harry
#False
```

## Loic Program Engine(LPE)
 - Unification
 - Backtracking

**Unification**
 - The process of finding **substitutions** for variables in logical expressions
   - such that the expression become equal and compatible
 - For example, a LPE can unify the terms cat(A), and cat(mary) by binding variable A to atom mary that means we are giving the value mary to variable A
 - A LPE can unify person(Kevin, Dane) and person(L, S) by binding L and S to stom Kevin and Dane, repectively
 - Sometimes it is called binding the **variables to values**

**Backtracking**
 - In the process of backtracking, a LPE will go back to the previous goal, and after that, it will try to find another wey to satisfy the goal
 - If the current colution fails, backtracking allows for exploring alternative solutions
 - E.g., 
   - 1\. The system starts bt attempting to satisfy the first goal or query
   - 2\. If the goal is satisfied, it continues to the next goal or query
   - 3\. If the goal fails, the system backtracks to the previous choice point and explores alternative choices
   - The system then continuesthe seach from the alternatice choice point, either trying adifferent posibility or exploring a different branch
   - If all possibilities and branches have been exhausted, and no more solutions are found, the search terminates

**An illustration**
 - When it finds that D is not the destination, it backtracks to B, then go to E, and backtracks again to B and then A, ...
 - When it finds G(the goal), it stops
![](/Lab4/Picture1.png)

 - X = {Q, NSW, V, T, SA, WA, NT}
 - We would follow a chronological order for assignments
![](/Lab4/Picture2.png)
   - Q = Red
   - NSW = Green
   - V = Blue
   - T = Red
   - SA = no legal value
 - Solution:
   - The backjumping method backtranks to the most recent assignment in the confilct set
   - Goto:T(assume we know T is not relevant)
   - Goto: V
   - Try other values for V
     - E.g. Red

## Logic Programming with Python
 - Kanren(minikanren): express the logic in terms of rules and facts
 - SymPy(sympy): a Python library for symbolic mathematics

```py
""from kanren import run, var, membero
from kanren import Relation, facts, lall
from kanren,contraints import neq, isinstanceo
from numbers inport Intrgral
from unification,match import *
""
x = var()
run(1, x, eq(x,5))
z = var()
run(1, x, eq(x,z), eq(z, 3))
From kanren import Ralation, facts
facts(parent, ("David","Bobby"),
              ("David", "Lisa"),
              ("Amy", "Davuid"))


run(1, x, parent, "Bobby")
run(2, x, Parent, "David")

run(0, grandparent(x, "David"))

y = var()
run(0, [x,y], grandparent(x, y))
```
**Relations: student**
```py
studies = Relation()
facts(studies, ("Charlie","CSC135"),
               ("Olivia", "CSC135"),
               ("Jack", "CSC131"),
               ("Arthur", "CSC134"))

teacher = Relation()
facts(teaches, ("Kirke", "CSC135"),
               ("Collins", "CSC131"),
               ("Collins", "CSC171"),
               ("Juniper", "CSC134"))

def professor(x,y):
  c = var()
  return lall(teaches(x,c), studies(y,c))

what = var()
run(0, what, studies('Charlie',what))

students = var()
run(0, students, profrssor('Kirke', students))






```         
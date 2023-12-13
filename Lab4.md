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
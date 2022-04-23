# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from collections import namedtuple

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



Node = namedtuple('Node', ['state', 'parent', 'action', 'cost'], defaults=(None,None,None,0))

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """
    state = problem.getStartState()
    first_node = Node(state, None, None)
    fringe = util.Stack()
    fringe.push(first_node)
    visited = []
    while not fringe.isEmpty():
        current = fringe.pop()
        if current.state in visited:
            continue
        visited.append(current.state)

        if problem.isGoalState(current.state):
            path = []
            node = current
            while node.parent:
                path.insert(0, node.action)
                node = node.parent
            return path

        for state_, action, cost in problem.getSuccessors(current.state):
            node = Node(state_, current, action)
            fringe.push(node)

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    first_node = Node(state, None, None)
    fringe = util.Queue()       # change to Queue only!
    fringe.push(first_node)
    visited = []
    while not fringe.isEmpty():
        current = fringe.pop()
        #print(current.state)
        if current.state in visited:
            continue
        visited.append(current.state)

        if problem.isGoalState(current.state):
            path = []
            node = current
            #print("Forming path...")
            while node.parent:
                #print(node.state)
                path.insert(0, node.action)
                node = node.parent
            return path

        for state_, action, cost in problem.getSuccessors(current.state):
            node = Node(state_, current, action)
            fringe.push(node)

    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    first_node = Node(state, None, None)
    fringe = util.PriorityQueue()           # use PriorityQueue
    fringe.push(first_node, 0)              # cost is 0
    visited = []
    while not fringe.isEmpty():
        current = fringe.pop()
        if current.state in visited:
            continue
        visited.append(current.state)

        if problem.isGoalState(current.state):
            path = []
            node = current
            while node.parent:
                path.insert(0, node.action)
                node = node.parent
            return path

        for state_, action, cost in problem.getSuccessors(current.state):
            node = Node(state_, current, action, cost=current.cost + cost)      # Node with cost
            fringe.push(node, node.cost)                                        # push with cost

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0




def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    first_node = Node(state, None, None)
    fringe = util.PriorityQueue()           
    fringe.push(first_node, heuristic(state, problem))                          # cost is h(n)
    visited = []
    while not fringe.isEmpty():
        current = fringe.pop()
        if current.state in visited:
            continue
        visited.append(current.state)

        if problem.isGoalState(current.state):
            path = []
            node = current
            while node.parent:
                path.insert(0, node.action)
                node = node.parent
            return path

        for state_, action, cost in problem.getSuccessors(current.state):
            node = Node(state_, current, action, cost=current.cost + cost)      # node cost = length of path to the node
            fringe.push(node, node.cost + heuristic(node.state, problem))       # priority  = node cost + heuristic cost                                    

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

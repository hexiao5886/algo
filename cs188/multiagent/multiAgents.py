# multiAgents.py
# --------------
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


from charset_normalizer import utils
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from math import inf
from numpy import mean

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def _evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def around(pos):
            """Return left, right, up, down and center of pos."""
            x, y = pos
            delta = [(0,1), (1,0), (-1,0), (0,-1), (0,0)]

            return [(x+dx, y+dy) for dx, dy in delta]

        score = 0
        
        for ghost in [0, 1]:
            ghostPos = newGhostStates[ghost].getPosition()
            if newScaredTimes[ghost] == 0 and newPos in around(ghostPos):
                return -inf                            # away from ghost

        x, y = newPos
        if currentGameState.getFood()[x][y] == True:    # must be current state food map
            score += 100                                 # eat food

        cx, cy = currentGameState.getPacmanPosition()
        foods = newFood.asList()
        if foods:
            nearestFood = min(foods, key=lambda xy: manhattanDistance(xy, newPos))
            fx, fy = nearestFood
            if (fx-cx)*(x-cx) + (fy-cy)*(y-cy) > 0:
                score += 5                              # go for food
        # sometimes it may dump into dead loop, add global food average pos to improve
        # One idea is to use MazeDistance but not manhattanDistance

        return successorGameState.getScore()

    def evaluationFunction(self, currentGameState, action):
        # reference: https://github.com/molson194/Artificial-Intelligence-Berkeley-CS188/blob/master/Project-2/multiAgents.py
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float("inf")

        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float("-inf")

        foodDist = []
        for food in list(newFood.asList()):
            foodDist.append(util.manhattanDistance(food, newPos))

        foodSuccessor = 0
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodSuccessor = 300

        return successorGameState.getScore() - 5 * min(foodDist) + foodSuccessor

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.minValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.getValue(successor, currentDepth, 1))
        return v

    def minValue(self, gameState, currentDepth, agentIndex):
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents()-1:
                v = min(v, self.getValue(successor, currentDepth+1, 0))
            else:
                v = min(v, self.getValue(successor, currentDepth, agentIndex+1))

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    # https://github.com/molson194/Artificial-Intelligence-Berkeley-CS188/blob/master/Project-2/multiAgents.py
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxValue = float("-inf")
        a, b = float("-inf"), float("inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1, a, b)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
            a = max(a, maxValue)
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex, a, b):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth,a,b)
        else:
            return self.minValue(gameState,currentDepth,agentIndex,a,b)

    def maxValue(self, gameState, currentDepth, a, b):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.getValue(successor, currentDepth, 1, a, b))
            if v > b:           
                return v
            a = max(a, v)
        return v

    def minValue(self, gameState, currentDepth, agentIndex, a, b):
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents()-1:
                v = min(v, self.getValue(successor, currentDepth+1, 0, a, b))
            else:
                v = min(v, self.getValue(successor, currentDepth, agentIndex+1, a, b))
            if v < a:              # can not be <= !!!!, psudo code on the website is '<', on the notes is '<='.
                return v           # The RIGHT way is to use '<='
            """
IF use '<=' instead of '<', the result is:

*** FAIL: test_cases/q3/6-tied-root.test
***     Incorrect generated nodes for depth=3
***         Student generated nodes: A B max min1 min2
***         Correct generated nodes: A B C max min1 min2
***     Tree:
***         max
***        /   \
***     min1    min2
***      |      /  \
***      A      B   C
***     10     10   0
            """

            b = min(b, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.expValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.getValue(successor, currentDepth, 1))
        return v

    def expValue(self, gameState, currentDepth, agentIndex):
        all_values = []
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents()-1:
                v = self.getValue(successor, currentDepth+1, 0)
            else:
                v = self.getValue(successor, currentDepth, agentIndex+1)
            all_values.append(v)

        return sum(all_values)/len(all_values)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    dead_end = -100 if len(currentGameState.getLegalActions(0)) == 1 else 0
    numFood = currentGameState.getNumFood()
    x, y = currentGameState.getPacmanPosition()

    foods = currentGameState.getFood().asList()
    food_dist = 0
    d_nearest = 0
    if foods:
        for food in foods:
            food_dist += manhattanDistance(food, (x, y))

        nearestFood = min(foods, key=lambda x: manhattanDistance(x, currentGameState.getPacmanPosition()))
        d_nearest = manhattanDistance(nearestFood, (x, y))


    ghosts = currentGameState.getGhostStates()
    scared = 1 if ghosts[0].scaredTimer > 0 else -1
    ghost_near = 0
    for ghost in ghosts:
        if util.manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()) < 2:
            ghost_near = 1
    
    """
    Here are some method calls that might be useful when implementing minimax.

    gameState.getLegalActions(agentIndex):
    Returns a list of legal actions for an agent
    agentIndex=0 means Pacman, ghosts are >= 1

    gameState.generateSuccessor(agentIndex, action):
    Returns the successor game state after an agent takes an action

    gameState.getNumAgents():
    Returns the total number of agents in the game

    gameState.isWin():
    Returns whether or not the game state is a winning state

    gameState.isLose():
    Returns whether or not the game state is a losing state
    """
    return dead_end - numFood*200 - d_nearest*5 + scared*ghost_near*1000

# Abbreviation
better = betterEvaluationFunction

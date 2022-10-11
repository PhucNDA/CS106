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

from numpy import Inf
from pyparsing import alphas
from util import manhattanDistance
from game import Directions
import random
import util
import math
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [
            self.evaluationFunction(gameState, action) for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates
        ]
        # return successorGameState.getScore()
        "*** YOUR CODE HERE ***"
        # OLD CODE

        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")
        foodList = food.asList()
        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer
                                                             == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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

        # util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            alpha,beta=None,None
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ = minValue(state.generateSuccessor(0, action), 1, 1,alpha, beta)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth,alpha,beta):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1,alpha,beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1, depth,alpha,beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

                if alpha is not None and value<=alpha:
                    return value

                if beta is None:
                    beta = value
                else:
                    beta = min(value, beta)    
                

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth,alpha,beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1, depth,alpha,beta)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

                if beta is not None and value>=beta:
                    return value

                if alpha is None:
                    alpha = value
                else:
                    alpha = max(value, alpha)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)

        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def maxValue(state, depth=0):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            val = None
            legalActions = state.getLegalActions()
            successState = [state.generateSuccessor(0,x) for x in legalActions]
            num_of_agents=state.getNumAgents()
            for states in successState:
                if val is None:
                    val=minValue(states, depth,num_of_agents -1)
                else:
                    val = max(val, minValue(states, depth, num_of_agents-1))
            return val

        def minValue(state, depth=0, idx=0):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(idx)
            successState = [state.generateSuccessor(idx, x) for x in legalActions]
            tmp = 0
            for s in successState:
                if idx > 1:
                    tmp += minValue(s, depth, idx-1)
                else:
                    tmp += maxValue(s, depth-1)
            return float(tmp)/len(successState)


        legalActions = gameState.getLegalActions()
        move = Directions.STOP
        val=-999999999
        num_of_agents=gameState.getNumAgents()
        for action in legalActions:
            tmp = minValue(gameState.generateSuccessor(0,action), self.depth, num_of_agents-1)
            if tmp > val:
                val = tmp
                move = action
        return move

def helper(table, pos):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    d = table.copy()
    for i in range(table.width):
        for j in range(table.height):
            d[i][j] = 1e5
    d[pos[0]][pos[1]] = 0
    q = [pos]
    while len(q) != 0:
        posu = q[0]
        q.pop(0)
        for i in range(4):
            x = posu[0] + dx[i]
            y = posu[1] + dy[i]
            if (x >= 0 and x <= table.width - 1 and y >= 0
                    and y <= table.height - 1 and d[x][y] == 1e5
                    and table[x][y] != '#'):
                q.append([x, y])
                d[x][y] = d[posu[0]][posu[1]] + 1

    return d


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    game_score = currentGameState.getScore()

    table = currentGameState.getWalls().copy()
    for i in range(table.width):
        for j in range(table.height):
            if (table[i][j] == True):
                table[i][j] = '#'
            else:
                table[i][j] = ' '
    food = currentGameState.getFood().copy()
    for i in range(table.width):
        for j in range(table.height):
            if (food[i][j] == True):
                table[i][j] = '.'
    table[newPos[0]][newPos[1]] = 'P'
    for ghost in newGhostStates:
        location = ghost.getPosition()
        table[int(location[0])][int(location[1])] = 'G'
    for caps in newCapsules:
        location = caps
        table[location[0]][location[1]] = '*'
    result = helper(table, newPos)

    # print(result)
    # print(table)

    # wall : #
    # food : .
    # pacm : P
    # ghos : G
    # caps : *
    # CODE MOI
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    check = 0
    for state in newGhostStates:
        dis=result[int(state.getPosition()[0])][int(state.getPosition()[1])]
        if(dis%2!=0):
            dis=1+dis//2
        else:
            dis=dis//2
        if state.scaredTimer != 0 and state.scaredTimer>=dis:
            check = 1
    if (check == 0):
        closestGhost = min([
            result[int(ghost.getPosition()[0])][int(ghost.getPosition()[1])]
            for ghost in newGhostStates
        ])
        num_cap = 1
        for caps in newCapsules:
            num_cap += 1
        if newCapsules:
            closestCapsule = min(
                [result[caps[0]][caps[1]] for caps in newCapsules])
        else:
            closestCapsule = 1
        foodList = newFood.asList()
        if foodList:
            closestFood = min([result[food[0]][food[1]] for food in foodList])
        else:
            closestFood = 1
        theta = 0
        if (closestGhost < 2):
            theta = 1e18

        return 200 * game_score + 10 / closestFood + 10 * len(
            foodList) + 50 / closestCapsule - theta
    else:
        closestGhost1 = 1e5
        count_bonus = 0
        for state in newGhostStates:
            dis=result[int(state.getPosition()[0])][int(state.getPosition()[1])]
            if(dis%2!=0):
                dis=1+dis//2
            else:
                dis=dis//2
            if state.scaredTimer != 0 and state.scaredTimer>=dis:
                closestGhost1 = result[int(ghost.getPosition()[0])][int(
                    ghost.getPosition()[1])]
                count_bonus += 1
        closestGhost2 = 1e5

        for state in newGhostStates:
            if state.scaredTimer == 0:
                closestGhost2 = min(
                    closestGhost2, result[int(ghost.getPosition()[0])][int(
                        ghost.getPosition()[1])])
        num_cap = 0
        for caps in newCapsules:
            num_cap += 1
        if newCapsules:
            closestCapsule = min(
                [result[caps[0]][caps[1]] for caps in newCapsules])
        else:
            closestCapsule = 1
        foodList = newFood.asList()
        if foodList:
            closestFood = min([result[food[0]][food[1]] for food in foodList])
        else:
            closestFood = 1
        theta = 0
        if (closestGhost2 < 2):
            theta = 1e18

        return 2000 * game_score + 50 / closestFood - 250 * closestGhost1 - theta + 50 * num_cap
    # CODE CU
    # closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    # if newCapsules:
    #     closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    # else:
    #     closestCapsule = 0

    # if closestCapsule:
    #     closest_capsule = -3 / closestCapsule
    # else:
    #     closest_capsule = 100

    # if closestGhost:
    #     ghost_distance = -2 / closestGhost
    # else:
    #     ghost_distance = -500

    # foodList = newFood.asList()
    # if foodList:
    #     closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    # else:
    #     closestFood = 0

    # return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule


# Abbreviation
better = betterEvaluationFunction

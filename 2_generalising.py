import random
import math

# In the previous tutorial, we hard-coded a few things, like our left-right finding and our memory states.
# However, what if we don't know what our initial states are going to be? Well, we can generalise the code to adapt to
# a variety of situations by dynamically allocating memory for possible actions.

# In this tutorial, we'll start with a 1D universe, then move on to a 2D universe. However, we'll structure the code so that we can
# quickly swap between the two universes, with no change to our original code base.

# While we're restructuring things, let's also make an agent class:


class reinforcementLearningAgent():
    def __init__(self, universeEngine):
        # Accept a universe engine in our constructor. The universe engine is responsible for providing a number of different things, as you will see in the following function declarations.
        self.universeEngine = universeEngine
        # This time, we will be using a dictionary of dictionaries instead of a list of dictionaries, so that we can quickly find our states even if they're
        # rather complex. For a 1D universe where we can simply use the indices of the array a list would have been fine, but for the 2d universe it would be more difficult.
        self.agentMemory = {}
        # We're also not going to explicitly initialise the agent memory this time. Instead, we'll add entries as we proceed through the map.
        self.discount = 0.5
        self.explorationChance = 1
        self.explorationDecayRate = 0.999
        # Our current location is now a string (as our state dictionary accepts strings, not integers.)
        self.currentLocation = universeEngine.getStartingState()
        # Finally, a few things to do with printing statistics
        self.reportAfterRounds = 100

    def makeMove(self):
        # Each time we make a move, we ask the universe engine what the possible moves are.
        moveList = self.universeEngine.generateMoves(self.currentLocation)
        # Now, we make sure that our memory remembers what states are allowed.
        if (self.currentLocation not in self.agentMemory):
            self.agentMemory[self.currentLocation] = {}

        # For this state, we also remember all the moves that are allowed, so that we can calculate values for them.
        for move in moveList:
            if (move not in self.agentMemory[self.currentLocation]):
                self.agentMemory[self.currentLocation][move] = 0

        # Make the move
        # Pick the best move out of the list, starting with the first move as a guess
        bestMove = moveList[0]
        bestMoveScore = self.agentMemory[self.currentLocation][bestMove]
        for move in moveList:
            if self.agentMemory[self.currentLocation][move] > bestMoveScore:
                bestMove = move
                bestMoveScore = self.agentMemory[self.currentLocation][move]

        # Make explorational moves as necessary
        if (random.random() < self.explorationChance):
            bestMove = random.randint(0, len(moveList)-1)
            bestMove = moveList[bestMove]

        # Now actually perform the reward
        (newLocation, immediateReward, isConcluded) = self.universeEngine.makeMove(
            self.currentLocation, bestMove)
        # Update based on the reward
        oldScore = self.agentMemory[self.currentLocation][bestMove]
        if newLocation in self.agentMemory:
            resultStateExpectedReward = max(
                self.agentMemory[newLocation].values())
        else:
            # we haven't been here before
            resultStateExpectedReward = 0

        # This is the magic formula again.
        self.agentMemory[self.currentLocation][bestMove] = immediateReward + \
            self.discount * resultStateExpectedReward

        self.currentLocation = newLocation

        return (isConcluded, immediateReward, bestMove)

    def runRound(self, printMoves):
        self.currentLocation = self.universeEngine.getStartingState()
        totalReward = 0
        isConcluded = False
        if printMoves:
            print("This round moves:")
        while not isConcluded:  # We haven't finished yet
            (isConcluded, immediateReward, currentMove) = self.makeMove()
            totalReward = totalReward + immediateReward
            if (printMoves):
                print("{}=>{}: {}".format(
                    currentMove, self.currentLocation, totalReward))
        return totalReward

    def runRounds(self, totalRounds):
        lastAverageScore = 0
        roundCount = 0
        self.explorationChance = 1
        # Currently, if we add more rounds, every round after round 1000 will have such a small exploration rate that
        # the bot will basically be doing the same thing over and over again. So instead, we can calulate the exploration decay rate so that it scales
        # proportionally to the number of rounds we have.
        # This gives us a 0.0001 exploration rate by the final round,
        self.explorationDecayRate = math.pow(0.0001, 1/totalRounds)
        # regardless of the number of rounds.
        while (roundCount < totalRounds):
            roundCount = roundCount+1
            shouldPrintMoves = False
            if roundCount % self.reportAfterRounds == 0:
                print("Current round: {}; Exploration chance was {}; Averge score was {}".format(
                    roundCount, self.explorationChance, lastAverageScore/self.reportAfterRounds))
                # print(self.agentMemory) # You can uncomment this line if you want to see the agent memory - but it gets pretty big!
                lastAverageScore = 0
                shouldPrintMoves = True
            lastAverageScore = lastAverageScore + \
                self.runRound(shouldPrintMoves)
            self.explorationChance = self.explorationChance * self.explorationDecayRate

# And that's it! This agent should be able to handle both our 1d game below and our 2d game below that.


# Let's now create a class for our 1D game. We need to setup the functions that we used in our reinforcement
# learning agent, namely getStartingState(), makeMove(location, move), and generateMoves(location)
class oneDimensionalUniverse():
    # We can set the tiles during initialisation.
    def __init__(self, tiles, startingTile, maxMoves):
        self.tiles = tiles
        self.startingTile = startingTile
        self.maxMoves = maxMoves

    def getStartingState(self):
        # One quirk of separating the engine and the agent is that the total moves allowed by the universe is now set by the universe. To convey this information,
        # we can add the turn count to the state.
        return "tile:{},move:{}".format(self.startingTile, 0)

    def generateMoves(self, location):
        # We only care about the tile part of the location when generating moves
        tile = location.split(",")[0]
        tile = int(tile.split(":")[1])
        if (tile == 0):
            return ["right"]
        elif (tile == len(self.tiles)-1):
            return ["left"]
        else:
            return ["left", "right"]

    def makeMove(self, location, action):
        location = location.split(",")
        tile = int(location[0].split(":")[1])
        move = int(location[1].split(":")[1])
        if action == "left":
            tile = tile-1
        else:
            tile = tile+1
        move = move+1
        newLocation = "tile:{},move:{}".format(tile, move)
        immediateReward = self.tiles[tile]
        isConcluded = (move == self.maxMoves)
        return (newLocation, immediateReward, isConcluded)


# And now to run our reinforcement learning agent against this universe!
if True:
    universe = oneDimensionalUniverse(
        [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10], 5, 10)
    agent = reinforcementLearningAgent(universe)
    agent.runRounds(10000)
# Go run it in the console and see what you get. (Hint: You can type `cls` or `clear` to clear the console so you can scroll back to where you started.)

# Let's try some other tests:
# universe = oneDimensionalUniverse([10, 0, 0, 0, 0, 1, 0], 3, 7) # This one's shorter! Oh no!
# universe = oneDimensionalUniverse([4, 7, 9, 8, -5, 1, 1, 2, 3, 4], 6, 11)

# And now to show the real power of our general agent: Let's extend this to 2D!
# We'll need another universe engine:


class twoDimensionalUniverse():
    def __init__(self, tiles, startingX, startingY, maxMoves):
        self.tiles = tiles  # This time tiles is a list of lists.
        self.startingX = startingX
        self.startingY = startingY
        self.maxMoves = maxMoves
        print(self.tiles)

    def getStartingState(self):
        # Again, tile is a pair of coordinates rather than a single coordinate
        return "tile:{}_{},move:{}".format(self.startingX, self.startingY, 0)

    def generateMoves(self, location):
        # We only care about the tile part of the location when generating moves
        tile = location.split(",")[0]
        tile = tile.split(":")[1]
        (currentX, currentY) = tile.split("_")
        currentX = int(currentX)
        currentY = int(currentY)
        possibleMoves = []
        if (currentX > 0):
            possibleMoves.append("left")
        if (currentX < len(self.tiles[0])-1):
            possibleMoves.append("right")
        if (currentY > 0):
            possibleMoves.append("up")
        if (currentY < len(self.tiles)-1):
            possibleMoves.append("down")
        return possibleMoves

    def makeMove(self, location, action):
        location = location.split(",")
        tile = location[0].split(":")[1]
        (currentX, currentY) = tile.split("_")
        currentX = int(currentX)
        currentY = int(currentY)
        move = int(location[1].split(":")[1])
        if action == "up":
            currentY = currentY-1
        elif action == "down":
            currentY = currentY+1
        elif action == "left":
            currentX = currentX-1
        elif action == "right":
            currentX = currentX+1
        move = move+1
        newLocation = "tile:{}_{},move:{}".format(currentX, currentY, move)
        immediateReward = self.tiles[currentY][currentX]
        isConcluded = (move == self.maxMoves)
        return (newLocation, immediateReward, isConcluded)


# Fantastic! We didn't even have to touch the agent, and now our agent can run around in a 2D universe.
# Let's try it out: (Disable by falsing-out the lines for the 1D universe above, and enable the lines for the 2D universe below)
if False:
    universe = twoDimensionalUniverse(
        [[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 10]], 2, 2, 10)
    agent = reinforcementLearningAgent(universe)
    agent.runRounds(10000)

############# Exercises ##############
# 1. Using the move records output, draw out the 2d grid above and see if the moves are indeed optimal.
# 2. Put in your own 2D grid. You can make a maze by having walls of -1000 score, paths of 0 score and target locations of 10 score. Does your agent solve the maze?
# 3. In the above 2D grid, estimate the number of possible pathways that the bot could have taken. Is 10,000 round a reasonable number of rounds? What if you only
# give the bot half the number of rounds?
# 3. [Challenge] Edit the code so that the universe engine is aware of the current move, but the agent is not. Do you still get the same results?
# 3. [Challenge] Create a new 1D universe engine with a few modifications: The agent has a limited amount of 'energy'; on each turn the agent can either move tiles (costing 1 energy) or
# harvest the current tile, which also costs 1 energy but gives the score of the current tile. Does the agent still generalise well?

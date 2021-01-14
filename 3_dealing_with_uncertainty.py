import random
import math

# In the previous tutorials, every move led to another state with 100% certainty. However, what can we do in cases where we aren't as certain about what the
# reward is, or where we will end up after each action?

# To find out, we need to look at two things: Expected value; and the Q-Learning equation.

# Say we have a random reward function, e.g. a dice roll:


def randomDiceReward():
    return random.randint(1, 6)
# You may remember from high school that the expected value of a random reward is the sum of ([the probability of a reward] times [the value of the reward]).
# So for this dice roll, the expected value is 1/6 * 1 + 1/6 * 2 + 1/6 * 3 + 1/6 * 4 + 1/6 * 5 + 1/6 * 6 = 3.5.

# But what if we have a much more complicated random reward function? Say:


def randomHardReward():
    num = random.random()
    nln = (num + math.log(abs(num)))
    return math.sin(abs(math.fmod(nln, math.pi)))
# Well, we could try and math it out. Or, we could get a computer to do it for us using trial and error:


def getExpectedValue(distribution, runs=10000):
    runsLeft = runs
    score = 0
    while runsLeft > 0:
        runsLeft = runsLeft-1
        score = score + distribution()
    return (score/runs)


# Let's try this out.
print(getExpectedValue(randomDiceReward))
# Did you get the result you wanted?
print(getExpectedValue(randomHardReward))

# With our reinforcement learning agent, if the reward of a particular tile is randomized, we can still use our many runs to determine the expected reward of a particular tile.

# There's another way we can improve the outcome of our reinforcement learning agent however, and it has the added bonus of making our learning even fasterer: The 
# Q-Learning equation! (Q stands for Quality, which is another word for Score that isn't really Score :3 go figure)

def bellmanActionScore(oldScore, immediateReward, expectedRewardFromNextState):
    discount = 0.5 # for example
    return immediateReward + discount * expectedRewardFromNextState

def qLearningActionScore(oldScore, immediateReward, expectedRewardFromNextState):
    alpha = 0.5 # for example
    discount = 0.5 # for example
    return oldScore + alpha * (immediateReward + discount * expectedRewardFromNextState - oldScore)

# The Q-learning equation modifies Bellman's equation by adding an extra hyperparameter called Alpha. Alpha controls how fast we learn: if we have 0 alpha we never learn, and 
# if we have 0.9+ alpha then we basically forget what we learnt last time. This is your second hyperparameter!

# Now, let's bring our general reinforcement learning agent back, and add the Q-learning with the alpha:


class qLearningAgent():
    def __init__(self, universeEngine):
        self.universeEngine = universeEngine
        self.agentMemory = {}
        # New hyperparameter! Yay!
        self.alpha = 0.5
        self.discount = 0.5
        self.explorationChance = 1
        self.explorationDecayRate = 0.999
        self.currentLocation = universeEngine.getStartingState()
        self.reportAfterRounds = 100

    def makeMove(self):
        moveList = self.universeEngine.generateMoves(self.currentLocation)
        if (self.currentLocation not in self.agentMemory):
            self.agentMemory[self.currentLocation] = {}
        for move in moveList:
            if (move not in self.agentMemory[self.currentLocation]):
                self.agentMemory[self.currentLocation][move] = 0
        bestMove = moveList[0]
        bestMoveScore = self.agentMemory[self.currentLocation][bestMove]
        for move in moveList:
            if self.agentMemory[self.currentLocation][move] > bestMoveScore:
                bestMove = move
                bestMoveScore = self.agentMemory[self.currentLocation][move]
        if (random.random() < self.explorationChance):
            bestMove = random.randint(0, len(moveList)-1)
            bestMove = moveList[bestMove]
        (newLocation, immediateReward, isConcluded) = self.universeEngine.makeMove(
            self.currentLocation, bestMove)
        oldScore = self.agentMemory[self.currentLocation][bestMove]
        if newLocation in self.agentMemory:
            resultStateExpectedReward = max(
                self.agentMemory[newLocation].values())
        else:
            resultStateExpectedReward = 0

        # This update formula has changed as well.
        self.agentMemory[self.currentLocation][bestMove] = oldScore + self.alpha * \
            (immediateReward + self.discount *
             resultStateExpectedReward - oldScore) 

        self.currentLocation = newLocation

        return (isConcluded, immediateReward, bestMove)

    def runRound(self, printMoves):
        self.currentLocation = self.universeEngine.getStartingState()
        totalReward = 0
        isConcluded = False
        if printMoves:
            print("This round moves:")
        while not isConcluded:
            # We're adding a few bits and bobs here so that we can see the agent's value of our last move.
            previousLocation = self.currentLocation
            (isConcluded, immediateReward, lastMove) = self.makeMove()
            totalReward = totalReward + immediateReward
            if (printMoves):
                print("{}=>{}: {}; agent value is {}".format(
                    lastMove, self.currentLocation, totalReward, self.agentMemory[previousLocation][lastMove]))
        return totalReward

    def runRounds(self, totalRounds):
        lastAverageScore = 0
        roundCount = 0
        self.explorationChance = 1
        self.explorationDecayRate = math.pow(0.0001, 1/totalRounds)
        while (roundCount < totalRounds):
            roundCount = roundCount+1
            shouldPrintMoves = False
            if roundCount % self.reportAfterRounds == 0:
                # We'll also do one new thing here: Count the number of states we've encountered so far.
                statesSeen=0
                for state in self.agentMemory:
                    statesSeen= statesSeen + len(self.agentMemory[state])
                print("Current round: {}; Exploration chance was {}; Averge score was {}; total states seen: {}".format(
                    roundCount, self.explorationChance, lastAverageScore/self.reportAfterRounds, statesSeen))
                lastAverageScore = 0
                shouldPrintMoves = True
            lastAverageScore = lastAverageScore + \
                self.runRound(shouldPrintMoves)
            self.explorationChance = self.explorationChance * self.explorationDecayRate

# Now, we'll modify our 1D universe so that instead of returning a fixed reward for each tile, each tile now gives a random reward from 0 up to the base value of the tile.
class oneDimensionalRandomUniverse():
    def __init__(self, tiles, startingTile, maxMoves, randomizer):
        self.tiles = tiles
        self.startingTile = startingTile
        self.maxMoves = maxMoves
        # Accept a randomizer function that generates a random reward based on the tile values.
        # We do this instead of hard coding the random function so we can quickly change the random
        # function down the line.
        self.randomizer = randomizer

    def getStartingState(self):
        return "tile:{},move:{}".format(self.startingTile, 0)

    def generateMoves(self, location):
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
        # We change the reward to be random here.
        immediateReward = self.randomizer(self.tiles[tile])
        isConcluded = (move == self.maxMoves)
        return (newLocation, immediateReward, isConcluded)


# Let's give it a go! Change this False to True to run this chunk of code.
if False:
    def linearRandomReward(base):
        return random.random()*base
    universe = oneDimensionalRandomUniverse(
        [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10], 5, 10, linearRandomReward)
    agent = qLearningAgent(universe)
    agent.runRounds(1000)
    # You'll probably see that the agent still performs the same set of actions (moving to the end and then hopping forwards and backwards), even if the reward was rather random.
    # Try and mess around with the number of run-rounds here, you might notice that when leaning with less rounds, the average score is lower. Pay attention to the 
    # moves the bot makes, this is due to it not having enough time to explore the extremidy of the map and score that juicy 10 points.

# Let's try something even more gruelling: This time, we either get no reward 50% of the time, or full reward the other 50% of the time.
if False:
    def binaryRandomReward(base):
        return int(random.random() > 0.5)*base
    universe = oneDimensionalRandomUniverse(
        [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10], 5, 10, binaryRandomReward)
    agent = qLearningAgent(universe)
    agent.runRounds(1000)
    # You'll probably see that the agent still performs the same set of actions...

# What if we only give the reward 5% of the time and give nothing the remaining 95% of the time?
if False:
    def harshBinaryRandomReward(base):
        return int(random.random() > 0.95)*base
    universe = oneDimensionalRandomUniverse(
        [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10], 5, 10, harshBinaryRandomReward)
    agent = qLearningAgent(universe)
    agent.runRounds(1000)
    # Interesting! My version of the bot has decided that it wants to collect the 1 reward it was standing on (by moving left and
    # then right) before moving on to the end. I personally would not have thought of that strategy, so there you go.

# Now, having a random reward is all fine, but what if we have a random state transition?
# In this next universe, there's a 20% chance that the agent will trip up and actually _not_ make the move it wanted to (but instead stay in its original square):


class trippyOneDimensionalUniverse():
    def __init__(self, tiles, startingTile, maxMoves):
        self.tiles = tiles
        self.startingTile = startingTile
        self.maxMoves = maxMoves

    def getStartingState(self):
        return "tile:{},move:{}".format(self.startingTile, 0)

    def generateMoves(self, location):
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
        if random.random()<0.5:
            if action == "left":
                tile = tile-1
            else:
                tile = tile+1
        else:
            # oops, tripped over
            pass
        move = move+1
        newLocation = "tile:{},move:{}".format(tile, move)
        # Let's make the reward non-random this time.
        immediateReward = self.tiles[tile]
        isConcluded = (move == self.maxMoves)
        return (newLocation, immediateReward, isConcluded)

# Can our agent hold up against this trippyness?
if False:
    universe = trippyOneDimensionalUniverse(
        [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10], 5, 10)
    agent = qLearningAgent(universe)
    agent.runRounds(1000)
    # The agent did alright; but notice how the total number of states went up from 80 to 130something. 

# Now, design wise, there is a lesson to be learnt here. An [ACTION] is different to a [STATE TRANSITION], because in our example a 
# single [ACTION] (moving left or right) has led to different possible [STATE]s.

# When you're designing your own RL agents, you might be tempted to make actions equal to state transitions, but this does not generalise
# well to circumstances where states can vary with the same action. 

# Mathematically, keeping actions and states apart means that each action has an expected value assigned to it, rather than each state. 
# Remember, we're designing an agent that chooses an action at any given point in time, not an agent that decides what state it's going to 
# be in in the next point in time (because it can't reliably predict that!)

############# Exercises ##############
# 1. In the original random universe, reduce the reportAfterRounds to 10, and take a look at the expected score for each run. Do the agent values ever converge?
# 2. Change alpha to 0.7, 0.5 and 0.2, and run the q-learning for only 100 rounds. Which one learns better?
# 3. With an alpha of 0.5, compare the current Q-learning agent with the previous bellman-based reinforcement learning agent. Which is better after 20 rounds? 50 rounds? 100 rounds? 
# 3. [Challenge] Take the 2d universe engine from the previous tutorial and add a probabilistic element to it. Compare the result with the original 2d universe engine.
# 4. [Challenge] To drive convergence, we can turn down the alpha parameter over time to reduce the amount of learning that occurs. Modify the code so that the alpha parameter starts 
# at 0.7 and decays to 0.01 by the last round. Is the system more convergent? Does it perform better over 100 rounds? 500 rounds?
# 5. [Challenge] Create an RL agent that is based on [STATE TRANSITIONS] rather than [ACTIONS]. Yes, I just told you not to. But give it a try anyway :3
import random
import math

# So far, we've only played with universes with a single agent in them. However, in real life, games might have multiple agents playing at the same time.
# Does geinforcement learning generalise to these kinds of situations?

# Let's create a universe engine for a tic-tac-toe game (except no diagonal wins! Because I don't want to bore you to death with the check-if-win-function):


class ticTacToeUniverse():
    def __init__(self, boardSize=3, winCondition=3):
        self.boardSize = boardSize
        self.winCondition = winCondition
        self.resetBoard()
        self.reportFrequency = 10
        self.whoseTurn = 0
        # Here, we'll set the score for winning to be 10.
        self.winReward = 10 
        # Let's also give the agents some encouragement to play:
        self.perTurnImmediateReward = 1

    def resetBoard(self):
        self.tiles = []
        for i in range(self.boardSize):
            row = []
            for j in range(self.boardSize):
                row.append("*")  # Let's have * = blank; o = circle; x = cross.
            self.tiles.append(row)

    def serialiseState(self):
        # This helper function helps turn the board state into something that can be stored as a string.
        output = ""
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                output = output+self.tiles[i][j]
        return output

    def prettyPrintState(self):
        # This function creates a human-friendly representation of the board.
        rows = []
        for i in range(self.boardSize):
            row = "|".join(self.tiles[i])
            row = row + "\n"
            rows.append(row)
        return "-+-+-\n".join(rows)

    def getWinner(self):
        winnerCandidate = "*"
        winnerCandidateCount = 0
        # Check horizontal wins
        for i in range(self.boardSize):
            winnerCandidate="*"
            winnerCandidateCount=0
            for j in range(self.boardSize):
                if not self.tiles[i][j] == "*":
                    if (winnerCandidate == self.tiles[i][j] or winnerCandidate == "*"):
                        winnerCandidate = self.tiles[i][j]
                        winnerCandidateCount = winnerCandidateCount+1
                    elif not winnerCandidate == "*":
                        winnerCandidate = self.tiles[i][j]
                        winnerCandidateCount = 1
                    else:
                        winnerCandidate = "*"
                        winnerCandidateCount = 0
                    if winnerCandidateCount == self.winCondition:
                        return winnerCandidate
        # Check vertical wins
        for i in range(self.boardSize):
            winnerCandidate="*"
            winnerCandidateCount=0
            for j in range(self.boardSize):
                if not self.tiles[j][i] == "*":
                    if (winnerCandidate == self.tiles[j][i] or winnerCandidate == "*"):
                        winnerCandidate = self.tiles[j][i]
                        winnerCandidateCount = winnerCandidateCount+1
                    elif not winnerCandidate == "*":
                        winnerCandidate = self.tiles[j][i]
                        winnerCandidateCount = 1
                    else:
                        winnerCandidate = "*"
                        winnerCandidateCount = 0
                    if winnerCandidateCount == self.winCondition:
                        return winnerCandidate
        # Check if the board is full
        if len(self.generateMoves()) == 0:
            return "#"
        else:
            # No winners yet
            return "*"
    # Now, before we proceed, we should note that previously, we had the agent driving the universe. This time, however, because there will be multiple
    # agents, we will let the universe drive the agents.
    # The agents will still be in charge of keeping their own memories intact, however.

    def registerAgents(self, agent1, agent2):
        self.agents = [agent1, agent2]
        agent1.registerUniverse(self)
        agent2.registerUniverse(self)

    def generateMoves(self):
        # Count the number of free tiles on the board, starting from the top left, and send them as a list.
        # The moves don't have to have any meaning to us humans - as long as the universe engine can unambiguously interpret what the move should be.
        # In this case, by numbering the free tiles from the top left, we can unambiguously figure out what move we want to take next, whilst keeping the
        # move representation to a single character.
        freeSquareCount = 0
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.tiles[i][j] == '*':
                    freeSquareCount = freeSquareCount+1
        # returns e.g. [1,2,3,4,5] if there are 5 free squares.
        return list(range(freeSquareCount))

    def makeMove(self, move):
        move = int(move)  # it's probably a string at this point
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.tiles[i][j] == '*':
                    if move == 0:
                        if self.whoseTurn == 1:
                            self.tiles[i][j] = "x"
                        else:
                            self.tiles[i][j] = "o"
                        return
                    else:
                        move = move-1
            # There is an additional quirk to newLocation however: since the new location is dependent on the other player, we can't immediately give the agent its
            # new state. Instead, we will separate the update function to an agent.updateOutcome() function.

    def runRound(self, shouldPrint):
        self.resetBoard()
        self.whoseTurn = 0
        while self.getWinner() == "*":
            # Now we need to juggle two agents at a time. We'll let one agent make their move, and then update the opposite agent based on their previous action.
            self.agents[self.whoseTurn].makeMove(self.serialiseState())
            self.whoseTurn = 1-self.whoseTurn
            # whoseTurn is now pointing to the opposite agent; update them.
            if (self.getWinner() == "x"):
                self.agents[1].updateState(
                    self.winReward, self.serialiseState())
                self.agents[0].updateState(-self.winReward,
                                           self.serialiseState())
            elif (self.getWinner() == "o"):
                self.agents[0].updateState(
                    self.winReward, self.serialiseState())
                self.agents[1].updateState(-self.winReward,
                                           self.serialiseState())
            elif (self.getWinner() == "#"):
                # on draw, update both agents
                self.agents[0].updateState(
                    self.perTurnImmediateReward, self.serialiseState())
                self.agents[0].updateState(
                    self.perTurnImmediateReward, self.serialiseState())
            else:
                self.agents[self.whoseTurn].updateState(
                    self.perTurnImmediateReward, self.serialiseState())

            if shouldPrint:
                print(self.prettyPrintState())

        self.agents[0].completeRound()
        self.agents[1].completeRound()

    def runRounds(self, totalRounds):
        # Set the exploration decay rate for both agents
        reportAfterRounds = int(totalRounds / self.reportFrequency)
        self.agents[0].explorationDecayRate = math.pow(
            0.0001, 1/totalRounds)
        self.agents[1].explorationDecayRate = math.pow(
            0.0001, 1/totalRounds)
        for roundCount in range(totalRounds):
            shouldPrint = (roundCount %
                           reportAfterRounds == 0) and (roundCount > 0)
            if shouldPrint:
                print("Currently at round {}".format(roundCount))
                self.agents[0].printStats()
                self.agents[1].printStats()
            self.runRound(shouldPrint)


# Now for the agents: We'll have three agents: A reinforcement learning agent, a random agent, and a human-interface agent that will let you play through the
# console window.


class reinforcementLearningAgent():
    def __init__(self):
        self.universeEngine = None  # We'll set this later
        self.agentMemory = {}
        self.alpha = 0.5
        self.discount = 0.5
        self.explorationChance = 1
        self.explorationDecayRate = 0.999
        # This time we're doing our own updates, so we need to keep track of our last location and last move.
        self.lastLocation = None
        self.lastMove = None
        # We do need to keep track of our own stats tho.
        self.roundScore = 0
        self.roundsElapsed = 0
        self.lastAverageScore = 0

    def registerUniverse(self, universeEngine):
        self.universeEngine = universeEngine

    def makeMove(self, currentLocation):
        moveList = self.universeEngine.generateMoves()
        if (currentLocation not in self.agentMemory):
            self.agentMemory[currentLocation] = {}
        for move in moveList:
            if (move not in self.agentMemory[currentLocation]):
                self.agentMemory[currentLocation][move] = 0
        bestMove = moveList[0]
        bestMoveScore = self.agentMemory[currentLocation][bestMove]
        for move in moveList:
            if self.agentMemory[currentLocation][move] > bestMoveScore:
                bestMove = move
                bestMoveScore = self.agentMemory[currentLocation][move]
        if (random.random() < self.explorationChance):
            bestMove = random.randint(0, len(moveList)-1)
            bestMove = moveList[bestMove]
        self.universeEngine.makeMove(bestMove)
        self.lastLocation = currentLocation
        self.lastMove = bestMove
        # We've simplified things a bit here because we don't do the update along with the move anymore.

    def updateState(self, immediateReward, newLocation):
        # Instead, the rest of the update steps go here...
        if self.lastLocation is None:
            return  # The second player gets a stray updateState on the first move, ignore it
        oldScore = self.agentMemory[self.lastLocation][self.lastMove]
        if newLocation in self.agentMemory:
            resultStateExpectedReward = max(
                self.agentMemory[newLocation].values())
        else:
            resultStateExpectedReward = 0
        self.agentMemory[self.lastLocation][self.lastMove] = oldScore + self.alpha * \
            (immediateReward + self.discount *
             resultStateExpectedReward - oldScore)
        self.roundScore = self.roundScore + immediateReward

    def completeRound(self):
        # do some cleaning up, etc
        self.explorationChance = self.explorationChance * self.explorationDecayRate
        self.lastAverageScore = self.lastAverageScore+self.roundScore
        self.roundsElapsed = self.roundsElapsed+1
        self.roundScore = 0
        self.lastLocation = None

    def printStats(self):
        statesSeen = 0
        for state in self.agentMemory:
            statesSeen = statesSeen + len(self.agentMemory[state])
        print("Exploration chance was {}; Averge score was {}; total states seen: {}".format(
            self.explorationChance, self.lastAverageScore/self.roundsElapsed, statesSeen))
        self.roundsElapsed = 0
        self.lastAverageScore = 0


# Alright, let's play the reinforcement agents against each other!
if True:
    universe = ticTacToeUniverse(3, 3)
    agent1 = reinforcementLearningAgent()
    agent2 = reinforcementLearningAgent()
    universe.registerAgents(agent1, agent2)
    universe.runRounds(100)
    # Hmm, it looks like there's a significant first-move-advantage, in this strange variant with no diagonal wins.
# To really see if these agents are any good however, we need to pitch them against a
# master tic-tac-toe player: You!

# Let's first create an agent that allows you to play:
class consoleInterface():
    def __init__(self):
        self.universeEngine = None  # We'll set this later

    def registerUniverse(self, universeEngine):
        self.universeEngine = universeEngine

    def makeMove(self, currentLocation):
        moveList = self.universeEngine.generateMoves()
        # Deserialise and pretty print the current location so that you, dear human, can see what is 
        # going on.
        board=list(currentLocation)
        boardGrid=[]
        emptySquares=0
        for i in range(self.universeEngine.boardSize):
            row=[]
            for j in range(self.universeEngine.boardSize):
                currentCell=board[i*3+j]
                if currentCell!="*":
                    row.append(currentCell)
                else: 
                    row.append(str(emptySquares+1)) # Use 1 based indexing so that we don't accidentally mix up 0 and o
                    emptySquares = emptySquares + 1
            boardGrid.append("|".join(row)+"\n")
        print ("Your turn to play! Enter the number corresponding to the grid cell you will play in.")
        print ("-+-+-\n".join(boardGrid))
        playerMove = int(input()) - 1
        self.universeEngine.makeMove(playerMove)

    def updateState(self, immediateReward, newLocation):
        # Here because otherwise the code will complain. You, human, have no obligation to change your ways...
        pass

    def completeRound(self):
        print (self.universeEngine.prettyPrintState())
        print ("Winner was {}".format(self.universeEngine.getWinner()))
        pass

    def printStats(self):
        # Here because otherwise the code will complain. Hopefully you know what's going on.
        pass

# Let's now train our agent and get it to play!
if False:
    universe = ticTacToeUniverse(3, 3)
    agent1 = reinforcementLearningAgent()
    agent2 = reinforcementLearningAgent()
    universe.registerAgents(agent1, agent2)
    universe.runRounds(100)
    playerInterface = consoleInterface()
    universe.registerAgents(agent1, playerInterface)
    universe.runRounds(1)
    # Run this two or three times. I can guarantee that if you're paying attention you can win at least once - the agent will make a mistake.
# Looks like the agents are pretty terrible after all... But why?
# Let's train the agents some more, with 1,000 rounds of training:
if False:
    universe = ticTacToeUniverse(3, 3)
    agent1 = reinforcementLearningAgent()
    agent2 = reinforcementLearningAgent()
    universe.registerAgents(agent1, agent2)
    universe.runRounds(1000)
    playerInterface = consoleInterface()
    universe.registerAgents(agent1, playerInterface)
    universe.runRounds(1)
    # Again, run this two or three times. The agent will make mistakes.
# Remember earlier when we set the immediate reward of making a move to be 1? Well, turns out this encouraged the agents to favour longer games 
# instead of just winning! Try setting the immediate reward to 0 and re-running the 100 round game. The agent should be pretty perfect now.

# This teaches us a lesson: A agent is only as good as its reward function -- and indeed, as good as the agent it's training against!
# To prove this, let's go and make a dumb random bot to train our agents against:

# Let's first create an agent that allows you to play:
class randomBot():
    def __init__(self):
        self.universeEngine = None  # We'll set this later

    def registerUniverse(self, universeEngine):
        self.universeEngine = universeEngine

    def makeMove(self, currentLocation):
        moveList = self.universeEngine.generateMoves()
        randomMove=moveList[random.randint(0,len(moveList)-1)]
        self.universeEngine.makeMove(randomMove)

    def updateState(self, immediateReward, newLocation):
        pass

    def completeRound(self):
        pass

    def printStats(self):
        pass

# Now, let's train an agent against this randombot, and then play against it as a human:
if False:
    universe = ticTacToeUniverse(3, 3)
    agent1 = reinforcementLearningAgent()
    agent2 = randomBot()
    universe.registerAgents(agent1, agent2)
    universe.runRounds(100)
    playerInterface = consoleInterface()
    universe.registerAgents(agent1, playerInterface)
    universe.runRounds(1)
    # Again, try a few times. You should be able to beat the agent.

# Fortunately, we can still get a pretty good agent by playing an agent against itself - thus allowing us to train agents against themselves.

############# Exercises ##############
# 1. Does the reinforcement learning algorithm fit a 4x4 grid with 4-in-a-row to win? Play with the parameters and see what you can come up with.
# 2. Play a randomly-trained agent against an [agent vs agent]-trained agent a few times. Which one wins?
# 3. Create a system for loading and saving the agentMemory to a file. If you load an agentMemory into an agent, will it still work? (Hint: consider using the `json` 
# library and some googling.)
# 4. [Challenge] Modify the ticTacToeUniverse class so that the runRounds() function takes a `validation` parameter that doesn't update the agentMemory; 
# then redo exercise 2 to get a statistically robust win-loss rate. (Note: If the exploration rate of the agents is still set to 0.01 at this point, the agents will play)
# the same game every time around! Can you find a way around this?)
# 5. [Challenge] One way of judging how well a agent plays is creating a 'perfect' agent, and then adding a 'corruption' factor, which determines the chance that the perfect
# agent will pick a random move instead of a good move. For example, a 10% corrupt agent plays random moves 10% of the time. Assuming a perfect agent is a agent that has been 
# trained 100,000 times against itself (Hint: save it to a file for consecutive runs!), create a system that judges agents based on this percent corruption score. Then,
# evaluate agents that have been trained only 10, 100, and 1000 times, using the corrpution metric.
# 6. [Challenge] A connect-4 game engine has been provided for you in the resources/connect4.py folder. Use it to create a connect4 bot.
# (Hint: A regular connect 4 game that is 6x7 has on the order of 2^42 = 4 trillion possible gamestates. You might want to go with a few less states first.)
# 7. [Challenge] One flaw of this reinforcement learner is that it tends to go for the first option first because bestMove starts with move 0 when picking a move. Correct this.

import math
import random
import re


class connect4GameEngine():
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.resetBoard()

        self.reportFrequency = 10
        self.whoseTurn = 0
        self.winReward = 10
        self.perTurnImmediateReward = 0

        # Use the power of regular expressions to determine wins
        # If this is black magic to you and you want to learn it, go to regexr.com. But it's not easy!
        self.winRegexes = []
        self.winRegexes.append(re.compile(
            "([^\\*\\|])\\1\\1\\1"))  # row win condition
        # column win condition
        self.winRegexes.append(re.compile("([^\\*\\|])"+("."*w+"\\1")*3))
        # forward diagonal win condition
        self.winRegexes.append(re.compile("([^\\*\\|])"+("."*(w+1)+"\\1")*3))
        # backward diagonal win condition
        self.winRegexes.append(re.compile("([^\\*\\|])"+("."*(w-1)+"\\1")*3))

    def resetBoard(self):
        self.board = []
        for i in range(self.h):
            self.board.append(['*']*self.w)  # make a w by h board of zeros

    def serialiseState(self):
        return "|".join(("".join(str(cell) for cell in row)) for row in self.board)

    def prettyPrintState(self):
        return "\n".join(("|".join(str(cell) for cell in row)) for row in self.board) +"\n\n"

    def generateMoves(self):
        # translate rows into columns
        validMoves = []
        for i, col in enumerate(self.board[0]):
            if col == '*':
                validMoves.append(i)
        return validMoves

    def makeMove(self, move):
        for i in range(len(self.board)-1, -1, -1):
            if self.board[i][move] == '*':
                self.board[i][move] = ('x' if self.whoseTurn else 'o')
                return True # move was ok
        return False # move was bad

    def getWinner(self):
        serial = self.serialiseState()
        for r in self.winRegexes:
            p = r.search(serial)
            if p is not None:
                return p.group(1)
        # also check if the game is a tie
        if serial.find("*") == -1:
            return "#"  # a tie
        return "*"

    def registerAgents(self, agent1, agent2):
        self.agents = [agent1, agent2]
        agent1.registerUniverse(self)
        agent2.registerUniverse(self)

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

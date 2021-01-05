import random

# (1/25) Imagine there's a robotic agent in a room with some treasure. (In a real system, the 'room' is all the possible states
# that a robot could be in; and the treasure is the value of the states.) Given the treasure map, can we have the agent
# learn what to do? Can we make the agent learn how to adapt to uncertainty, or even other agents?

# (2/25) Let's start with the basics. Here's a treasure map which is a long line, with some rooms having a score in them.
# the agent could hover around the middle for a low score or go for the ends...
tileScores = [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10]

# (3/25) The agent starts at the centre where the 1 is, and the agent has
agentMovesTotal = 6
# moves in total. Every time the agent moves, it will get the score in the room it moves to. (It can revisit rooms.)
# The agent can't hang around in a room however, it has to move each time.


# (4/25) How do we get the agent to learn? Well, we can tell the agent to run the map a lot of times and remember what
# actions have what outcome. Let's start by giving the agent a memory:
agentMemory = []
# (5/25) What shall we store in the memory? Well, for each room, we have a few actions: move left, or move right. We can
# store the expected outcome of either moving left or right.

# (6/25) But wait! If the agent starts from the centre and moves left, then it will get 0 score immediately, but if it keeps moving
# it could get up to 10 score. So the score for the tile to the left should be more than 0, because there is a reward to be
# gained from moving left.

# (7/25) So, for each tile, and each possible action, we remember an 'expected reward'. For this simple game, we can layout all the states:
agentMemory = [
    {
        'right': 0
    },  # leftmost tile
    {
        'left': 0,
        'right': 0
    },  # second-from-leftmost tile
    {
        'left': 0,
        'right': 0
    },  # etc
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0,
        'right': 0
    },
    {
        'left': 0
    }  # rightmost tile
]
# (8/25) Where this is impractical, as we will see in future, we can add states as they appear. But that's a later problem (Part 2).
# We've currently got all the scores set to 0, which is fine, because they'll be updated as our agent experiments with its
# surroundings.

# (9/25) How do we calculate the score of each tile? Well, if an action (e.g. moving into a room with score 5) gives us reward, we
# factor in the reward. If an action takes us to a state which eventually leads to a reward, then we need to take that into
# account also.


def calculateNewActionScore(oldScore, immediateReward, expectedRewardFromNextState):
    # (10/25) This is known as Bellman's equation. We're not using oldScore at the moment, but it'll come in handy in part 3.
    return immediateReward + discount * expectedRewardFromNextState


# (11/25) What is discount? discount sets the cost of getting a future reward, compared to getting a reward immediately, because we'll
# take some turns to do so. We call disocunt a hyperparameter, because it affects how our other parameters (stored in agentMemory) will change.
discount = 0.5

# (12/25) You can try playing around with this value to see how your bot behaves. Discount should stay between 0 (super instant gratification mode) and 
# 1 (what does short term safety even mean). 

# (13/25) There's one more thing to do here before we move on: How do we calculate the next state reward (expectedRewardFromNextState)? Well, once
# we get to a state, we will take the action within that state that gives us maximum reward. So, the value of the next state can
# be deduced as the maximum reward from the actions available at the next state.


def calculateExpectedRewardFromNextState(state):
    if ('left' not in state):
        return state['right']
    elif ('right' not in state):
        return state['left']
    else:
        return max(state['left'], state['right'])


# (14/25) Alright, let's now create a function that tells the agent where to go next.
def basicWhereToNext(state):
    if ('left' not in state):
        return 'right'
    elif ('right' not in state):
        return 'left'
    else:
        if (state['left'] > state['right']):
            return 'left'
        else:
            return 'right'

# (15/25) But hold on. If our agent moves perfectly each time, how will it know to explore the board?
# We need to tell it to make bad decisions some of the time, in order to allow it to explore a bit. However, we also want it to eventually
# settle on making better decisions so it actually does what we want.


# (16/25)Let's create another hyperparameter set that determines the probability that the bot will make a questionable choice instead of a perfect one:
explorationChance = 1
# And let's have this chance slowly reduce by a factor:
explorationDecayRate = 0.999
# In this case, after every cycle we have a 100% - 99.9% = 0.1% reduction in our exploration chance, until eventually our exploration chance
#  goes to zero.

# (17/25) Our new movement algorithm now looks like this:
def whereToNext(state):
    if ('left' not in state):
        return 'right'
    elif ('right' not in state):
        return 'left'
    else:
        if (random.random() < explorationChance):
            # explore by taking a random move
            if (random.random() > 0.5):
                return 'left'
            else:
                return 'right'
        else:
            # don't have any fun, just do what we're meant to
            if (state['left'] > state['right']):
                return 'left'
            else:
                return 'right'

# (18/25) Almost there! We now need to set up the maze, so to speak. We'll have a number of rounds, and during each round the agent will explore
# the maze and update itself, and show us how it's going.
def runRound():
    # start at the centre. Remember, lists start from 0, so the centre #6 is actually 5.
    agentPosition = 5
    agentMovesRemaining = agentMovesTotal  # give the agent 6 moves to navigate
    totalScore = 0
    while (agentMovesRemaining > 0):
        agentMovesRemaining = agentMovesRemaining-1
        # (19/25) Decide where to go
        currentState = agentMemory[agentPosition]
        agentAction = whereToNext(currentState)
        # actually move the agent
        if (agentAction == 'left'):
            agentNewPosition = agentPosition - 1
        else:
            agentNewPosition = agentPosition + 1
        # (20/25) Update our agent by updating the expected score of the move we just did
        oldScore = agentMemory[agentPosition][agentAction]

        resultStateImmediateReward = tileScores[agentNewPosition]

        resultStateExpectedReward = calculateExpectedRewardFromNextState(
            agentMemory[agentNewPosition])

        agentMemory[agentPosition][agentAction] = calculateNewActionScore(
            oldScore, resultStateImmediateReward, resultStateExpectedReward)

        # update where we are
        agentPosition = agentNewPosition
        totalScore = totalScore + resultStateImmediateReward

    # When we're done, report the total score we earned
    return totalScore


# (21/25) Finally, we run a lot of rounds and get the agent to display how it's going at the end of every set of 50 rounds:
totalRounds = 5000
reportAfterRounds = 100
currentRound = 0

# (22/25) Let's also keep track of the average score over the past 50 rounds - this should get better over time:
lastAverageScore = 0

while currentRound <= totalRounds:
    currentRound = currentRound+1
    score = runRound()
    lastAverageScore = lastAverageScore + score
    # Explore less next time
    explorationChance = explorationChance * explorationDecayRate

    # Remember, this is the modulo operator, which gives the remainder of the first number divided by the second.
    if currentRound % reportAfterRounds == 0:
        # Show us what is going on
        print("Current round: {}; exploration percentage: {}%".format(currentRound, explorationChance*100))
        print(agentMemory)
        print("Average score is {}".format(
            lastAverageScore / reportAfterRounds))
        # reset the last average score
        lastAverageScore = 0

# (23/25) And that's it! Run the program by opening a console (you can press Control/command + shift + C if you're in vscode) and typing in "python 1_reinforcement_101.py". 
# Here's a few questions for you:
# 1. How fast did the agent memory converge? (How many rounds before the agentMemory was basically the same?)
# 2. Did you notice the score increasing between rounds?

# (24/25) Finally, play around with the hyperparameters a bit and see what happens.

# Now you might be thinking - Hey, I could have just programmed my robot to move left, and it would have
# gotten the same result much faster!

# (25/25) Well, here's the thing. You can copy and paste in any of the maps below and you don't have to change
# anything - the robot will re-learn what to do and do it. That's the power of reinforcement learning!

# tileScores=[5,4,3,2,1,0,1,2,3,4,5] # encourage the agent to move towards the ends
# tileScores=[0,0,3,0,0,1,0,0,3,0,0] # a smart agent shouldn't stray too far
# tileScores=[10,-1,-1,-1,0,-1,-1,-1,10] # some pain before you get to the reward
# tileScores=[10,0,0,0,0,0,0,0,0,0] # Assymetry? Oh no!

############# Exercises ##############
# 1. Come up with a tile set of your own, and try it out.
# 2. Try out alpha values of 5, 0.5, 0.05. Which one converges fastest?
# 3. Try out discount values of 5, 0.5, 0.05. Which ones still work well?
# 4. Try out exploration decay rates of 0.99, 0.999, 0.9999 (default was 0.999). Do we fully explore every pathway each time? Is 
# the end result reliable enough to use each time?
# 5. [Challenge] Create a function that calculates the number of rounds until convergence is achieved.
# 6. [Challenge] Using the function above, create a hyperparameter tuner that finds the optimal hyperparameters for a given tileset.
# Please do not (auto-)format the file so all the imports are at the top. That will make it more confusing to read.
import random

# Heavy kudos to https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd.
# But I take the credit for the on point explanations (hopefully?) :P and also the preivous tutorials were all written from scratch.

# In the previous tutorials, we've only dealt with environments with discrete states. However, in the real world outside of games, robots
# operate in environments with complex, continuous states. How can we overcome this?

# If we abstract the agent a bit, we find that the agent's memory informs the agent's decisions about what to do. If we abstract it
# further, our agent has a function that takes in a list of moves, as well as reinforcement-learned information about the current
# state, and then produces an outcome. It also needs a function that updates the reinforcement-learned information.

# One useful way of coming up with black box functions that update themselves is using a neural network. If you're not sure what a
# neural network is, it might be good to watch a few videos now, or pester the USRC execs about coming up with a neural network tutorial.
# For now, you need to know the following things:
# 1. A neural network consists of a bunch of units called neurons. Each neuron typically takes a bunch of input values from other neurons and
# gives off one single output value, which is utilised by a bunch of other neurons.
# 2. Neural networks have a bunch of parameters in them, which determine how the neuron behaves. When we train a neural network, we update these
# parameters until the neural network is doing what we want.
# 3. We can determine how well the neural network is doing by using a loss function; and then using some smart maths we can use the loss function to
# update the neural network to minimise the loss function. Let's call this "Loss Minimisation" to avoid overloading your vocabulary (but you may already
# know it / may see it elsewhere as "Back Propagation").
# 4. Overall, the neural network can be used as a magical black-box function which we can use to do cool stuff.

# So if overall our neural network creates a magic black-box function, what function do we want to create?
# If we go back to our update formula from q-learning, we had:
"""
newScore = oldScore + alpha * (immediateReward + discount * max(next state action scores) - oldScore)
"""
# It might be tempting to replace this equation with our neural net now, but the equation is fine - let's leave it alone for now. Instead, we
# also need to recognise that newScore depends on the state and the action associated with that state. So technically, what we're doing is:
"""
def oldScore(state,action):
    return agentMemory[state][action]

newScore(currentState, currentAction) = oldScore (currentState, currentAction) + alpha * (immediateReward + discount * max (next state action scores) - oldScore(currentState, currentAction))
"""

# Then, if it's impractical to just store every state because there are too many states, we can use a neural net as a black box function to estimate
# the score of a given state and action:
"""
def oldScore(state,action):
    return neuralNetworkEstimateOf(state,action)

(desired) newScore(state, action) = oldScore (state, action) + alpha * (immediateReward + discount * max (next state action scores) - oldScore(state, action))
"""

# We're getting closer to the answer, but we still have two more complications to sort out.

# (1/2) We previously stored newScore in our agentMemory; but we
# aren't using agentMemory anymore. So how can we store newScore? The answer is to use another neural network (let's call it newNeuralNetwork):
"""
def newScore(state,action):
    return newNeuralNetworkEstimateOf(state,action)
"""
# Then, instead of explicitly setting the newScore (because we can't just set the output of a neural network), we need to update the newScore using the q-learning result. Remember,
# when updating a neural network, we need to minimise the loss function which is the difference between what we have and what we want, so we should actually have:
"""
loss = newScore (state, action) - desiredNewScore(state,action)
"""
# If we now substitute in the formula for the desired new score, we have:
"""
loss = newScore (state, action) - (oldScore (state, action) + alpha * (immediateReward + discount * max (next state action scores) - oldScore(state, action)))
"""
# Now, because maths, we need the loss function to always be positive, so we square it:
"""
loss = (newScore (state, action) - (oldScore (state, action) + alpha * (immediateReward + discount * max (next state action scores) - oldScore(state, action))))^2
"""
# Finally, one variation you may see is that we do multiple trials at a time, in batches, because then we can use that expensive af GPU that performs parallel calculations that
# one does not simply afford for fun in 2020.
# At the end of every chunk of batches, we also swap over the old neural net and the new one so that we can make some headway.
# Again: THE OLD NEURAL NET IS A COPY OF THE NEW NEURAL NET WHICH IS UPDATED EVERY SO OFTEN. Don't say I didn't tell you.
"""
loss = Expected value over multiple runs (newScore (state, action) - (oldScore (state, action) + alpha * (immediateReward + discount * max (next state action scores) - oldScore(state, action))))^2
"""
# This is a mouthful, so this is often abbreviated to
"""
L(w) = E[(Q(s,a;w) + a(r + d * max (Q(s', a'; w')) - Q(s,a;w)) - Q(s,a;w))^2]
L(w) = E[(a(r + d * max (Q(s', a'; w')) - Q(s,a;w)))^2]
where
L = loss
E = expected value

w = parameters (aka weights) of your neural network
Q = magic neural net function
w' = oldScore version of magic neural network function; and its weights. So Q(s,a;w) is the new neural net, and Q(s,a;w') is the old neural net.

s = current state
a = current action
s', a' = future state; future state's prospective actions

r = reward
d = discount rate
a = alpha
"""
# (2/2) Aight, we're 90 lines in and no code, so you know you're in for a big one. If you wanna go take a break, that's fair game.

# We now need to think about this max (next state action scores) business. In our fancy equation above, that is written as max (Q(s',a'; w')), i.e. the
# different states and actions that the current action could lead to. But wait! We mentioned that the whole point of this exercise was that there were too many
# states and actions to store; so how can we list every possible action and state that is available here? What if we don't know what state the current action will
# even result in?

# The best we can do for now is to limit either s' or a'; and then feed the neural network the other one (a' or s') based on experimentation. If we base our
# neural net on a continuous state and have discrete actions, then we can limit our a' options and keep our s' continuous based on our observations. This is more
# commonly done because experiments tend to have plenty of states and fewer actions; but perhaps theoretically you could do it the other
# way around.

# Finally, some code! There are 5 main steps to this exercise:
# 1. Create the environment
# 2. Create the agent
# 3. Run a set of experiments
# 4. Train the agent based on the experiments
# 5. Repeat until complete

# Go install pytorch, by running `pip install torch`. Careful with your internet connection btw, this is pretty big.
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np  # If you haven't used numpy before, it contains a bunch of optimised functions for doing operations on large sets of numbers, making it super
# useful for computer vision, statistics, and neural networks.

################## 1. Create the environment ###############
# Also install the openAI gym `pip install gym`, which contains a bunch of fun machine learning tasks, of which we will be using a stick-balancing exercise:
import gym
import resources.mountainCar
env = gym.make('Acrobot-v1')

numStateVariables = env.observation_space.shape[0]
numActions = env.action_space.n

print('Number of state features: {}'.format(numStateVariables))
print('Number of possible actions: {}'.format(numActions))

# If you're lucky enough to afford a GPU, it's time to put it to use.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################## 2. Create the agent #####################
# First let's go for something familiar: our hyperparameters!
alpha = 1
# time moves forward irrespective of what we do, might as well aim for as long as we can.
discount = 0.99

# Set the exploration chance:
explorationChance = 1
# But also let's funk it up a bit! Instead of aiming for a tiny exploration at the end, let's aim for a target exploration rate of:
finalExplorationChance = 0.05
# Otherwise the agent will probably just do nothing, because nothing is safe.


def updateHyperParameters():
    global explorationChance
    if explorationChance > finalExplorationChance:
        explorationChance -= 0.001

# Let's set up the neural network:


class multilayerNeuralNetwork(nn.Module):
    def __init__(self, numStateVariables, numActions):
        super(multilayerNeuralNetwork, self).__init__()
        # Initialise a three-layer neural network with 32 hidden nodes in the middle. Three layers turns out to be enough for most applications in
        # neural networks.
        self.inputLayer = nn.Linear(numStateVariables, 32)
        self.middleLayer = nn.Linear(32, 32)
        self.outputLayer = nn.Linear(32, numActions)
        # I'd put this in an array but pytorch is very picky about how layers must be defined.

    def forward(self, states):
        # A relu is a type of activation function which defines a type of neuron in our neural network. Don't worry too much about it for now.
        x = F.relu(self.inputLayer(states))
        x = F.relu(self.middleLayer(x))
        return self.outputLayer(x)


# Now we'll make two of them for the old and new neural nets.
newNeuralNetwork = multilayerNeuralNetwork(
    numStateVariables, numActions).to(device)
oldNeuralNetwork = multilayerNeuralNetwork(
    numStateVariables, numActions).to(device)


def chooseAction(currentState, explorationChance):
    result = random.random()
    if result < explorationChance:
        # Random action from the environment sample space (in this case, left or right).
        return env.action_space.sample()
    else:
        actionScores = newNeuralNetwork(currentState).cpu().data.numpy()
        # Pick the best action that our neural network has suggested
        return np.argmax(actionScores)

# Now that we have the neural network, we can also set up the training system.


# We'll use the pre-built ADAM optimiser to train our network.
# An optimiser is a pre-built tool that helps update our neural network using our chosen update function.
# Different optimisers have different advantages; but ADAM is a good general-purpose optimiser that represented
# a big leap forward compared to others when it was introduced in 2015. It's pretty maths heavy though, so if
# you want to figure out how it works, you'll probably want to set aside another day or so.
optimizer = torch.optim.Adam(newNeuralNetwork.parameters(), lr=1e-4)

# We can also use a mean-square-error loss function, which is short for E((difference)^2).
lossFunction = nn.MSELoss()
# Note that this means we'll have to turn
"""
L(w) = E[(a(r + d * max (Q(s', a'; w')) - Q(s,a;w)))^2]
"""
# into
"""
L(w) = MeanSquareError((a(r + d * max (Q(s', a'; w')),
    Q(s,a;w))
"""
# i.e. splitting it into two arguments for the mean square error function.

# When we train, we'll need to feed the trainer the states, actions, rewards, next states and done-flags of our training data.
# This is somewhat more convoluted than our train-as-you-go model, but the training is made GPU-friendly by keeping the training
# runs the same length and using a done-flag to signify the end of the round, because GPUs like same-length arrays.


def performTraining(states, actions, rewards, nextStates, dones):
    # Get the old neural network to generate what it believes are the best action scores for each state.
    bestOldActionScores = oldNeuralNetwork(nextStates).max(-1).values
    # Use our Q-learning formula to calculate our target values that we'll use for the mean square error.
    target = alpha * (rewards + (1.0 - dones) * discount * bestOldActionScores)
    # This gives us the first argument of our mean square error function above, i.e. (a(r + d * max (Q(s', a'; w'))

    # Now on to the second argument:
    bestNewActionScores = newNeuralNetwork(states)
    # We only want to take the action scores corresponding to the actions that we actually did though, so we do this:
    """
    actuallyUsedActionScores=[]
    for (index, action) in enumerate(actions):
        actuallyUsedActionScores.append(bestNewActionScores[index][action])
    """
    # A faster, GPU friendly way of doing the same thing is as below:
    actionsMask = F.one_hot(actions, numActions)
    actuallyUsedActionScores = (actionsMask * bestNewActionScores).sum(dim=-1)
    # If you want to figure out what this means, do search up one-hot functions, as they come up quite a bit in
    # discrete machine learning.
    # the detach() is there to save your RAM, because
    loss = lossFunction(actuallyUsedActionScores, target.detach())
    # target is attached to the original neural net it came from, which we're about to abandon anyway.

    # Now, run the training function!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Return the loss so we can see how well we're doing.
    return loss


################ 3. Run the experiments ################
totalRounds = 1000
# Cache the samples so we can train after each round, instead of training during the middle of the round.
sampleCache = {
    "states": [],
    "actions": [],
    "rewards": [],
    "nextStates": [],
    "dones": []
}
if False:
    for currentRound in range(totalRounds):
        currentState = env.reset().astype(np.float32)  # reset the universe
        thisRoundReward = 0
        done = False
        while not done:
            # Save the currentState early since it is about to be type-converted
            sampleCache["states"].append(currentState)
            # Choose an action
            currentState = torch.from_numpy(
                np.expand_dims(currentState, axis=0)).to(device)
            action = chooseAction(currentState, explorationChance)
            # Run the simulation
            (nextState, reward, done, info) = env.step(action)
            if currentRound % 100 == 0:
                env.render()
                # only draw every 100th attempt

            # Store and update everything
            thisRoundReward += reward
            sampleCache["actions"].append(action)
            sampleCache["rewards"].append(reward)
            sampleCache["nextStates"].append(nextState)
            sampleCache["dones"].append(done)
            currentState = nextState.astype(np.float32)
        # Round over, let's train!
        # Copy over the old new neural net to the old neural net.
        oldNeuralNetwork.load_state_dict(newNeuralNetwork.state_dict())
        # convert into a pytorch friendly data type
        for dataType in sampleCache:
            sampleCache[dataType] = np.array(sampleCache[dataType])
        # We also need to do some type conversions:
        sampleCache["dones"] = sampleCache["dones"].astype(np.float32) # This was a boolean, it needs to be a number instead
        sampleCache["nextStates"] = sampleCache["nextStates"].astype(np.float32) # This was a double, it needs to be a float instead or pytorch will complain
        sampleCache["states"] = sampleCache["states"].astype(np.float32) # This was a double, it needs to be a float instead or pytorch will complain
        sampleCache["actions"] = sampleCache["actions"].astype(np.int64) # This was a float, but we're using it as array indexes down the line, so we need it
        sampleCache["rewards"] = sampleCache["rewards"].astype(np.int64) # This was a float, but we're using it as array indexes down the line, so we need it
            # to be integer
        for dataType in sampleCache:
            sampleCache[dataType] = torch.as_tensor(
                sampleCache[dataType], device=device)
        loss = performTraining(sampleCache["states"], sampleCache["actions"],
                                sampleCache["rewards"], sampleCache["nextStates"], sampleCache["dones"])

        print("Round: {}; Loss so far: {}, Last round score: {}, exploration: {}".format(
            currentRound, loss, thisRoundReward,explorationChance))
        # empty the buffer
        sampleCache = {
            "states": [],
            "actions": [],
            "rewards": [],
            "nextStates": [],
            "dones": []
        }
        updateHyperParameters()
    env.close()  # finish up

# So it turns out that this version hasn't improved much even after 10,000 rounds. However, if we look at the output in the console, we notice that the loss is pretty low! This 
# suggests that the neural net is working in that it's doing exactly what we told it to do. Perhaps we should change our approach.

# The version below is slightly modified so that instead of just training on the last round's data, it trains on ALL collected data, and also trains at EVERY timestep in the 
# simulation. This gives it a lot more training time, which will hopefully produce better results.

# However, because we're training on all the data, feeding it all into our neural net would be too much; so we'll take a sample of 32 data points at a time instead:
sampleSize = 32

# We also don't want to be constantly copying the old neural net over to the new one, so let's set a variable which determines how often that happens:
copyOverAfter = 2000 # samples

totalRounds = 1000
# In the sample cache we'll use a deque, or a double-ended queue, with a fixed size. This allows us to discard old samples so we don't overflow our ram.
from collections import deque
maxCacheSize=10000
sampleCache = {
    "states": deque(maxlen=maxCacheSize),
    "actions": deque(maxlen=maxCacheSize),
    "rewards": deque(maxlen=maxCacheSize),
    "nextStates": deque(maxlen=maxCacheSize),
    "dones": deque(maxlen=maxCacheSize)
}
sampleCount=0

# Watch the rewards from the previous 100 rounds
last100RoundsRewards=[]
trainingsSoFar=0

if True:
    for currentRound in range(totalRounds):
        currentState = env.reset().astype(np.float32)  # reset the universe
        thisRoundReward = 0
        loss = None
        done = False
        while not done:
            # Save the currentState early since it is about to be type-converted
            sampleCache["states"].append(currentState)
            # Choose an action
            currentState = torch.from_numpy(
                np.expand_dims(currentState, axis=0)).to(device)
            action = chooseAction(currentState, explorationChance)
            # Run the simulation
            (nextState, reward, done, info) = env.step(action)
            if currentRound % 100 == 0:
                env.render()
                # only draw every 100th attempt

            # Store and update everything
            thisRoundReward += reward
            sampleCache["actions"].append(action)
            sampleCache["rewards"].append(reward)
            sampleCache["nextStates"].append(nextState)
            sampleCache["dones"].append(done)
            currentState = nextState.astype(np.float32)
            # This time, constantly train, as long as we have enough samples to train off (so don't train before the first 32 frames ever encountered).
            if len(sampleCache["states"])>sampleSize:
                # We have enough samples, let's start training
                ## Take a subset of the lifetime data to train off
                sampledIndexes = np.random.choice(len(sampleCache["states"]), sampleSize) # Pick some random indexes between 0 and len(sampleCache["states"]), e.g. [1,4,32,88]

                ## convert into a pytorch friendly data type, and sample the data using array indexing
                chosenSamples={}
                for dataType in sampleCache:
                    convertedData = np.array(sampleCache[dataType])
                    chosenSamples[dataType] = convertedData[sampledIndexes]
                    ## do the sampling using array indexing
                        # https://stackoverflow.com/questions/2621674/how-to-extract-elements-from-a-list-using-indices-in-python; Dmitri's answer
                
                chosenSamples["dones"] = chosenSamples["dones"].astype(np.float32) # This was a boolean, it needs to be a number instead
                chosenSamples["actions"] = chosenSamples["actions"].astype(np.int64) # This was a float, but we're using it as array indexes down the line, so we need it to be an int
                chosenSamples["nextStates"] = chosenSamples["nextStates"].astype(np.float32) # This was a double, but we need it to be a float so pytorch doesn complain
                chosenSamples["states"] = chosenSamples["states"].astype(np.float32) # This was a double, but we need it to be a float so pytorch doesn complain
                chosenSamples["rewards"] = chosenSamples["rewards"].astype(np.float32) # This was a double, but we need it to be a float so pytorch doesn complain

                for dataType in sampleCache:
                    chosenSamples[dataType] = torch.as_tensor(
                        chosenSamples[dataType], device=device)

                # Do the training!
                loss = performTraining(chosenSamples["states"], chosenSamples["actions"],
                                        chosenSamples["rewards"], chosenSamples["nextStates"], chosenSamples["dones"])
                trainingsSoFar+=1
            # Copy over the old new neural net to the old neural net.
            sampleCount = sampleCount + 1
            if sampleCount % copyOverAfter == 0:
                oldNeuralNetwork.load_state_dict(newNeuralNetwork.state_dict())

        if currentRound >= 100:
            last100RoundsRewards = last100RoundsRewards[1:]
        last100RoundsRewards.append(thisRoundReward)

        print("Round: {}; Loss so far: {}, 100-round running average: {}, exploration: {}, trainingsSoFar:{}".format(
            currentRound, loss, np.mean(last100RoundsRewards),explorationChance,trainingsSoFar))
        updateHyperParameters()
    env.close()  # finish up

# We see in this example that we have a lot more than 1000 trainings, even though we only did 1000 rounds total. This gives our neural network a lot more experience so it can perform better.

############# Exercises ##############
# 1. Let's list out all of the hyperparameters including the new ones to do with the deep neural network:
# - alpha
# - discount
# - exploration decay
# - [NEW] middle layer size(s)
# - [NEW] number of middle layers
# - [NEW] sample batch size for training
# - [NEW] cache of previous samples
# - [NEW] time between swapping old and new networks
# - [CHALLENGE] neuron types (relu vs others)
# Find these variables in the code and change them. How do the results compare?
# 2. In the `gym` module, there are lots of other RL challenges. One of them is called MountainCar-v0. Replace 'CartPole-v0' with 'MountainCar-v0' and run the training. 
# Looking at the results from the training, can you guess why the particular reward system is not a good one?
# 3. [CHALLENGE] The source code for MountainCar-v0 has been provided in resources/mountainCar. You should be able to replace `env=gym.make('MountainCar-v0')` with 
# `env = resources.mountainCar.MountainCarEnv()` and have this file's code run without any other modifications. Your task is to edit resources/mountainCar so that the 
# reinforcement learner is able to make at least some headway.
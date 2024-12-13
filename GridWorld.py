import numpy as np
import matplotlib.pyplot as plt


class GridWorld(object):
    """
    Initialize the shape of the Grid and the Magic Squares
    m=length (number of rows), n=width (number of columns)
    """
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n
        """
        this state space does not include the terminal state
        (which is the bottom right)
        """
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove(self.m * self.n-1)
        """
        the State Space + includes the terminal state
        """
        self.stateSpacePlus = [i for i in range(self.m * self.n)]
        """
        we need to know how the actions map to the changes in
        the environment
        """
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        """ 
        we need to keep track of the set of possible actions
        """
        self.possibleActions = ['U', 'D', 'L', 'R']
        """
        Add the magic squares
        """
        self.addMagicSquares(magicSquares)
        """
        Agent position
        """
        self.agentPosition = 0
    """
    How to add the magic squares
    """
    def addMagicSquares(self, magicSquares):
        """
        initialize the magic squares
        """
        self.magicSquares = magicSquares
        """
        let there be only 2 magic squares in this example
        """
        i = 2
        """
        iterate over the magic squares
        """
        for square in magicSquares:
            """
            position x is the floor of the current magic square
            and the number of rows.
            """
            x = square // self.m
            """
            position y is the modulus of the number of columns
            """
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            """
            Recall that the magic squares are a dictionary.
            We are iterating over the keys (source), and the values
            are the destination.
            """
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1
    """
    How to recognize the terminal state:
        We just find the state which is in the State Space 
            and not in the State Space Plus
    """
    def isTerminalState(self, state):
        s = self.stateSpace
        sp = self.stateSpacePlus
        return state in sp and state not in s
    """
    Get Agent row and column
    """
    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y
    """
    Set a new state
    """
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1  # this is how we represent the agent
    """
    What if the agent decided to move off the Grid?
    """
    def offGridMove(self, newState, oldState):
        # Already off Grid
        if newState not in self.stateSpacePlus:
            return True
        # Trying to move off Grid from the top or bottom
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        # Trying to move off Grid from the left or right
        elif oldState % self.m == self.m - 1 and newState % self.n == 0:
            return True
        else:
            return False
    """
    Now, we need a way to actually step:
    """
    def step(self, action):
        x, y = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

        if resultingState in self.magicSquares.keys():
            resultingState = self.magicSquares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0

        # if the agent is not trying to move offGrid:
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return (resultingState, reward,
                    self.isTerminalState(self.agentPosition), None)
        else:
            return (self.agentPosition, reward,
                    self.isTerminalState(self.agentPosition), None)
    """
    The grid needs to be reset at the end of every episode
    """
    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition
    """
    We have to provide a way for rendering
    """
    def render(self):
        print('-----------------------------')
        for row in self.grid:
            for column in row:
                if column == 0:  # if it is an empty square
                    print('-', end='\t')
                if column == 1:  # if there is an agent in the column
                    print('X', end='\t')
                if column == 2:  # this is an entrance to a magic square
                    print('A-in', end='\t')
                if column == 3:  # we just left the first magic square
                    print('A-out', end='\t')
                if column == 4:  # we are in another magic square
                    print('B-in', end='\t')
                if column == 5:  # we just left the second magic square
                    print('B-out', end='\t')

        # Print a new line after each row
        print('-----------------------------')
    """
    Define the Action Space Sample
    """
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)


"""
Max Action function
"""


def maxAction(Q, state, actions):
    """
    Create a numpy array of the agent's estimates of the
    present values of the expected future rewards
    """
    values = np.array([Q[state, a] for a in actions])
    # get the maximum action
    action = np.argmax(values)
    # return the actions that correspond to that action
    return actions[action]


"""
Now, we code a Q-Learning Agent and test the program
"""

# if '__name__' == '__main__':

"""
Now, we define the 2 magic squares as follows:
        {magicSquare_position1: send_forward_position, 
         magicSquare_position2: send_backward_position}
"""
magicSquares = {18: 64, 63: 19}
"""
We now create our 9 x 9 gridworld and specify the magic squares
"""
env = GridWorld(9, 9, magicSquares)
"""
Now we set the hyperparameters:
    How fast does the agent learn? - Learning rate (ALPHA)
    How does the agent value a potential future reward? 
        - GAMMA (A GAMMA of 1.0 means that the agent would be 
            totally far sighted (i.e., it would count all 
            future rewards equally.
    There is also an epsilon value (for epsilon-greedy action
                                    selection)
        - So learning would begin randomly, then it would converge
                on an epsilon greedy fashion.
"""
ALPHA = 0.1
GAMMA = 1.0
EPSILON = 1.0

"""
Initialize the Q-Learning table (dictionary)
"""
Q = {}
for state in env.stateSpacePlus:
    for action in env.possibleActions:
        """
        Zero (0), as used below, is an optimistic initial value.
        This means that since the agent receives a reward of -1 
        for each step, it can never have a reward of zero (because 
        there is some distance between the agent and the exit).
        By setting the initial state at zero, we encourage 
        the exploration of unexplored states.
        """
        Q[state, action] = 0

# Set number of Games, and rewards
number_of_games = 50000
total_rewards = np.zeros(number_of_games)

env.render()

for i in range(number_of_games):
    if i % number_of_games == 0:
        print('Starting Game', i)

    # resent the 'done' flag at the start of every episode
    done = False
    # begin every episode with zero rewards.
    # (do not accumulate rewards)
    episode_rewards = 0
    # reset the environment at the start of every new episode
    observation = env.reset()

    while not done:
        # take a random number for the epsilon-greedy
        # action selection
        rand = np.random.random()
        # find the maximum action for a given state
        action = maxAction(
            Q, observation,
            env.possibleActions) if rand < (
                1 - EPSILON) else env.actionSpaceSample()

        observation_, reward, done, info = env.step(action)

        episode_rewards += reward

        """
        Now, calculate the maximal action for the new state
            and insert the result into the update equation 
            for the Q-function
        """
        action_ = maxAction(Q, observation_, env.possibleActions)

        # Update the Q-function for the current action and state
        Q[observation, action] = (
            Q[observation, action] + ALPHA*(
                reward + GAMMA*Q[
                    observation_, action_] - Q[observation, action]))

        # let the agent know that the environment has changed states
        observation = observation_

    """
    At the end of every episode, we decrease epsilon so that 
         the agent settles on a purely greedy strategy.
    """
    if EPSILON - 2 / number_of_games > 0:
        # decrease EPSILON by 2 divided by number
        # of games every episode
        EPSILON -= 2 / number_of_games
    else:
        EPSILON = 0

    total_rewards[i] = episode_rewards

# plot the total rewards
plt.plot(total_rewards)
plt.show()

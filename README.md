# Cart-Pole Deep Q-Network
Goal: Train a Deep Q-Network that can balance a Cart-Pole.

# Summary
This project dealt with recreating the simple Cart Pole problem and using Reinforcement Learning solve it. To review, the Cart Pole prolem is about developing a solution to keep a pole upright that is attached to a cart with wheels. This has been already solved with classic control theory, which take the route of linearizing the problem and using linear controls such as LQR to calculate the force needed to push the cart with the required force to keep the pole upright. This problem can also be solved using Reinforcement Learning. In this approach a neural network is trained to learn a policy that can be used by an agent to select the right action based on the current states of the cart-pole.

For my project, I decided to add some differences to the orignal problem by attempting to solve the Cart Pole problem using a non-linearized system and using a runga-kutta 4th order numerical integrator. Another difference from the classic problem was that I choose a larger action space of four different forces to push the cart. They can be described as: Hard Left, Soft Left, Soft Right, Hard Right. The classic problem only pushes the cart left or right at a constant force. I still used the same observation space of cart position and velocity and pole angle and angular velocity to be inputted into my non-linearized state space differential equations.

Within this project I developed the envronment, inheriting from the gymnasium Environment class, to include my deisred observation space and action space. In it a step function was created to take in the states and action and use runge kutta to calcualte the next states. This information was used by the reward function to help the agent learn. 

Learning was done by a 3 layer neural network in combination with the bellman's equation. This lead to calculating Q values, which represent scores for the action taken at a particular state. It must also be mentioned that at each step the agent selected the action either through exploration, selecting a random action, or by using the learned policy that was generated with the neural network. The exploration represent the epsilon-greedy technique commonly used in these problems. It allows the agent to explore various action and find the best possilbe Q values.

The results from this project showed that the trained models were hitting rewards 


Cart Pole Performance
Model means: [1240.5, 1089.25, 1800.0]
Model stds: [914.24, 913.11, 337.44]
Overall mean: 1376.58, Overall std: 305.7

### NEED TO WRITE:
- How did I design my project, what files were created and why, the network architecture
- Results of my environment. Put some plots and evaluation metrics
- Stopping sooner that I want but give reason.

# What I Learned


# Future Work
- Add rendering to Cart Pole Environment to view animation.
- Create docker containerization.
- Find the best hyper-parameters.
- Compare trained policies against a basiline random policy.
- Improve visualization of plots.
- Add logging.
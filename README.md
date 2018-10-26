# LegsBenedict
A bipedal walker that is trained using reinforcement learning.

## Introduction
This project is my midterm project submission for the [Move 37 Course](https://www.theschool.ai/courses/move-37-course) from the [School of AI](https://www.theschool.ai/). The goal of this project is to train a bipedal robot to walk. The robot learns using reinforcement learning - a method of machine learning, where the learner (also called the agent) performs actions based on its current observed state in its environment, and is rewarded or penalized for its actions. Over many episodes of interaction with the environement, the agent attempts to maximize it's expected future reward, given its current state. The function that determines (or gives a probability for) the agent's action for each state is called the policy. The method of reinforcement learning I am using here is effectively several iterations of evaluating a policy, improving this policy, and re-evaluting it, until we have a policy that results in satisfactory behavior.

## Actor-Critic approach
The agent in this project is comprised of two learners. The "actor" is a neural network that predicts an action, given the current state of the agent. This is effectively a policy network. The "critic" is a neural network that takes the current state and the predicted action as an input, and evaluates the value of the state-action combination. Over many iterations, the actor and the critic both learn to get better at their roles, with the actor predicting actions that maximize expected reward and the critic evaluating the values of these actions more accurately. The paper this algorithm is based on is [Lillicrap(2015) - Continuous control with deep reinforcement learning].(https://arxiv.org/abs/1509.02971)

## The environment
The environment I am using to train my agent is the [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environment from OpenAI Gym. The robot is rewarded for moving forward for a total of 300 points up to the far end. If the robot falls, it gets -100 points. Applying motor torques also costs some points. The state (observable) consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements.

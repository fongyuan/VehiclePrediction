# Experimentation Log

## Feature Engineering
* Using just the positions for all agents as features
* Use PCA, etc to reduce the dimensionality of the feature space.
* Use 

Train one model for cars and one model for pedestrians

## Using the position of all agents (Auto-regression)

There are 11 pairs of (x,y) coordinates per agent. This leads to 22 features. Since there are at most 10 agents per scene, there could be a maximum of 220 features regarding position. 

In order to autoregress based on these features, we need to predict the next position for all agents at each timestep. I initally thought that since all agents exhibit the same behavior (they are all vehicles - will ignore pedestrians for now), I can train just two models for x and y position and use just these models to make a prediction for all agents.

It's dawning on me now that coordinate system is shifted such that the ego-vehicle's final position after a second is (0,0). This could lead to problems if only two models are trained on a relatively small feature set.

## Using the position of all agents (Non-autoregression) - 31 models lol

If we are auto-regressing, we can't fully utilize 3 seconds of positional information that is provided to us. Instead, what we can attempt is to build 31 models that each predict different timesteps on the same training data. So essentially, the same training data, different labels.


***

## Submissions
* submission #1 : linear regression with just agent position information
* submission #2 : Autoregressive. 22 featuressame with Lasso (L1)
* submission #3 : Non-autoregressive. 220 features with Lasso (L1)

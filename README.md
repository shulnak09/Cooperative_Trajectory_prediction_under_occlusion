# Cooperative_Trajectory_prediction_under_occlusion
This repo presents an algorithm to cooperatively estimate the state of an occluded object and probabilistically predict the future states.  The algorithm relies upon
relative pose estimation to recover the $\[\mathrm{R} | \mathrm{t}\]$: rotation and translation  between two sensors sharing common visual information. This relative 
pose is used to estimate occluded pedestrian's state from one sensor to another sensor's reference. The estimated states are passed though an approximate Bayesian 
neural network (BNN) which uses deep ensembles and Monte Carlo dropout to probabilistically predict future states.

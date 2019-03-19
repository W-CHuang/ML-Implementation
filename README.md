# ML-Implementation
This repository contains scripts that I write for implementations of some machine-learning algorithms with python
03.19 2019
1. Wrote a class to do gradient descent, which included the following methods:
* Batch Gradient Descent
  * Iteration
    * Use all training samples to update theta.
* Stochastic Gradient Descent
  * Iteration
    * loop (number of samples)
      * randomly choose one sample to update theta
* miniBatch Gradient Descent
  * Iteration
    * loop (number of batches)
      * use samples in batches to update theta
* Normal equation
  * Matrix Operation
To-do: should do convergence analysis to remove unneccessary iterations after convergence.

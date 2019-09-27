# UQ_Python

Code for Uncertainty Quantification sampling, certain functions ported over from legacy MATLAB code. For use in testing the performance differences between the following methods on the datasets:

<center>

| Dataset  | MCMC Method |
|----------|-------------|
| 2 Moons |   Gaussian Regression    |
| Gaussian Clusters  | Gibbs-Probit  |
| LAPD  | pCN-Probit  |
| HUJI  | pCN-BLS  |
| Gas Plume  |   |

</center>

## Uncertainty Quantification in Graph Based Semi-Supervised Learning
We approach the problem of semi-supervised learning for classification with a graph based approach for Bayesian inverse problems, in which we define a function on the nodes of a similarity graph representing the dataset and use that function to model the labeling of the underlying datapoints. Solving this inverse problem then reduces to minimizing a functional with respect to this node function, giving us a point estimate. Many machine learning algorithms likewise seek to find point estimates of corresponding posterior distributions for their respectively defined model. All models considered in this code base have posterior distributions that can be written in the form:

<p align="center"><img src="https://rawgit.com/millerk22/UQ_Python/master/svgs/5c9c626493e53ee49b58a58dfd447ea7.svg?invert_in_darkmode" align=middle width=374.80409999999995pt height=39.30498pt/></p>


We further seek to quantify our confidence or certainty in the our outputs. Appealing to the Bayesian nature of the models involved, we use samples from the posterior distributions corresponding to the respective models to characterize the uncertainty of our outputs. In this way, we obtain more than just a point estimate of the posterior distribution. These samples allow us to give not only predictions about the classification of unlabeled nodes, but also have some idea of confidence about such predictions.

## Active Learning -- Future Directions

Also, we will be testing "Human-in-the-Loop" active sampling schemes to progressively improve the predictions given by the schemes/models, but choosing points to ask a "human" (AKA "oracle") to give the model the labeling for. We will seek to test and develop more effective active sampling schemes for the datasets and models we investigate here.


## Running the Code

The necessary data structures representing the similarity graphs of the desired datasets are contained in ``Data_obj`` classes. For each of the datasets, there are corresponding functions ``load_*`` functions in the file ``datasets\dataloaders.py``. Then, the different MCMC sampling methods are defined by ``MCMC_Sampler`` objects, each kind corresponding to the MCMC method (e.g. ``Gibbs_Probit_Sampler`` object implements the Gibbs-Probit MCMC method). These sampler objects then can load ``Data_obj`` objects, and then run its respective sampling method with the loaded dataset, saving the desired mean points, statistics, etc. This object-oriented approach allows for simple, flexible use of the different datasets and methods.

To run the current test being worked on, run ``python basic_test.py`` in the command line in this current directory, which runs the Gibbs-Probit MCMC Sampler on a datset of 3 Gaussian Clusters. Can also use the following flags in your command line call:
* ``--embed 1`` to run IPython session with the saved variables after the test run
* ``--show 1`` to show plot of data set (only if data is 2-dimensional, currently)

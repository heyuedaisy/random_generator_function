# random_function_generator
This python file is used to generate random target functions and simulate dataset with specific signal and noise. 

# Background
Simulated datasets are usually employed to prove the quality of a newly put-forward algorithm, because dataset in the real world is hard to be obtained. Without doubt, features can be randomly generated from various distributions, such as Guassian distribution or uniform distribution, which can be easily realized in python. However, as for the target functions, it's difficult for us to define. Recently, inspired by rulefit,a kind of interpretable machine learning, I decide to apply this fantastic algorithm in various simulated datasets. After reading several classic papers by Friedman, I write this file to generate random target functions. Hope that this file can be helpful.

# Theories
One of the most important characteristics of any problem affecting performance is the true underlying target function F(x). Every method has particular targets for which it is most appropriate and others for which it is not. Since the nature of the target function can vary greatly over different problems, and is seldom known, we compare the merits of several learning ensemble methods on a variety of different randomly generated targets (Friedman 2001). 

Details can be found in the following papers.

# Reference
1. Importance Sampled Learning Ensembles,Jerome H. Friedman, Bogdan E. Popescu,2003.
2. Predictive Learning via Rule Ensemble, Jerome H. Friedman, Bogdan E. Popescu,2005.

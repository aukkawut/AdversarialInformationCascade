# Adversarial Information Cascade
This is the repository for my final project in OIE 559 Advanced Prescriptive Analytics at WPI. The work here is about the competitive cascading behavior of the 2 informations within the networks. The goal here is to maximize the expected edge spreading.

## Why not normal information cascade?

Consider the case where you want to talk about the politics. Let say there are two parties: A and B. A supporters can share the information of A to further support A or share the information of B to criticize them. Thus, looking at the node itself might not reflecting the behavior of the network.

## How is this network being generated?

The network is generated from Erdős–Rényi model.

## The goal

The goal here is to create an integer (or mixed integer) programming that solve the strategy where we want to maximize the expected spreading of the edges. Yes, this model is stochastic so we need to add "expected" as a prefix.

## Method

The method here is quite cheating if you asked me. The project is required you to use some sort of programming (as an optimization) but I use the library for doing the task. Anyways, this one exploit the concept from the stochastic programming in influence maximization [1]. Where there is no cost for picking the node so it would be purely stochastic (i.e., purely recourse) which I approximate the probability measure with the similar idea to the reinforcement learning: setup one policy (random policy) and explore the game to estimate the measure itself. Then, we use this to prune out the network. (i.e., remove edge that we are quite certain, given the finite simulation, that we will lose). Now, as suggested by [1], this problem now is the max flow problem. that is, we perform max flow after the pruning to get the network such that we have the high chance that we can spread to estimate the expected win.

```
[1] Wu, H.-H., and Küçükyavuz, S. A two-stage stochastic programming approach for influence maximization in social networks. Computational Optimization and Applications 69, 3 (Apr 2018), 563–595.
```

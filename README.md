# Metropolis-Hastings MCMC for A/B Testing 📊

This project provides a complete implementation of a **Metropolis-Hastings MCMC** (Markov Chain Monte Carlo) sampler from scratch. Its purpose is to demonstrate a **Bayesian approach** to A/B testing and compare it to a classical **frequentist approach**. The script is structured to be both a functional tool for A/B test analysis and a clear educational resource for understanding the core mechanics of MCMC.

***

## Project Overview

With this tool, you can:
* Simulate A/B test data with specified true conversion rates.
* Analyze the simulated data using a traditional **frequentist Z-test** for proportions.
* Implement and run a **Metropolis-Hastings algorithm** to perform a Bayesian analysis.
* Generate samples from the posterior distribution to compute key metrics like posterior means and credible intervals.
* Visualize the MCMC chain's convergence, posterior distributions, and the joint posterior to assess the results.
* Directly compare the conclusions drawn from both the frequentist and Bayesian methodologies.

***

## Comparison of Approaches

### Frequentist Approach
* **Method**: Uses a classical statistical **Z-test for proportions**.
* **Hypothesis**: Tests a null hypothesis ($H_0$) like $H_0: \text{rate}_B - \text{rate}_A \leq 0$ against an alternative hypothesis ($H_1$).
* **Output**: Provides a **p-value** and a binary decision to either accept or reject the null hypothesis.
* **Philosophy**: Assumes true parameters are fixed and unknown. The focus is on the probability of the data given the null hypothesis.
* **Interpretation**: "If the null hypothesis is true, what's the probability of seeing this data?"

### Bayesian Approach
* **Method**: Uses **Markov Chain Monte Carlo (MCMC)** to sample from the posterior distribution.
* **Formula**: Computes the posterior: $P(\theta_A, \theta_B \mid \text{data}) \propto P(\text{data} \mid \theta_A, \theta_B) \times P(\theta_A, \theta_B)$.
* **Output**: Provides **posterior means**, **credible intervals**, and direct probabilities like $P(\theta_B > \theta_A \mid \text{data})$.
* **Philosophy**: Treats parameters as random variables and focuses on the probability of the parameters given the observed data.
* **Interpretation**: "Given this data, what's the probability that $\theta_B$ is greater than $\theta_A$?"

***

## Key Concepts

* **Frequentist**: Employs hypothesis testing, p-values, and confidence intervals to make decisions.
* **Bayesian**: Focuses on posterior distributions, credible intervals, and direct probabilities.
* **MCMC**: A class of algorithms that use Markov Chains to sample from complex probability distributions.
* **Metropolis-Hastings**: A specific MCMC algorithm that uses an acceptance-rejection rule to construct a chain that converges to the target distribution.
* **A/B Testing**: The process of comparing two versions of a product or experience to determine which one performs better.

### The Value of the Bayesian Approach
The Bayesian approach offers a richer and more intuitive understanding of A/B test results compared to traditional frequentist methods.

* **Intuitive Probabilities**: Provides direct, actionable probabilities (e.g., "There is a 97% probability that Group B has a higher conversion rate than Group A"). This is often easier for non-statisticians to understand and act upon.
* **Uncertainty Quantification**: Bayesian credible intervals give a natural sense of parameter uncertainty (e.g., "there is a 95% probability that the true conversion rate lies within this range").
* **Incorporating Prior Knowledge**: It allows you to explicitly include prior information, which can be crucial for small sample sizes.

***

## Algorithm Overview

The Metropolis-Hastings algorithm is at the core of the Bayesian analysis in this project.

* **Target**: The goal is to sample from the posterior distribution $P(\theta_A, \theta_B \mid \text{data})$.
* **Method**: A Markov Chain is constructed that converges to this target distribution.
* **Proposal**: New parameter values are proposed using a "random walk" with a symmetric normal distribution centered on the current parameters.
* **Acceptance**: A new set of parameters is accepted or rejected based on the acceptance ratio: $\alpha = \min(1, \frac{\pi(\theta_{\text{proposed}})}{\pi(\theta_{\text{current}})})$, where $\pi$ is the unnormalized posterior.
* **Result**: The algorithm generates a set of samples that approximate the shape of the posterior distribution.

***

## Usage

To run the script and perform the A/B test analysis:

1.  **Dependencies**: Ensure you have the required libraries installed:
    ```bash
    pip install numpy matplotlib scipy statsmodels
    ```
2.  **Execution**: Run the Python script directly. The `main()` function is configured to:
    * Generate synthetic data.
    * Run the frequentist test.
    * Execute the Metropolis-Hastings MCMC simulation.
    * Analyze and visualize the results.
3.  **Configuration**: You can adjust key parameters within the `main()` function to explore different scenarios:
    * `true_rate_A`, `true_rate_B`: The true conversion rates for the simulation.
    * `n_A`, `n_B`: The number of visitors for each group.
    * `iterations`: The number of MCMC simulation steps.
    * `proposal_std`: The standard deviation for the MCMC random walk, which can be tuned to optimize the acceptance rate.

***

## Future Improvements

* **Dynamic Prior Specification**: Implement a user-friendly way to specify different prior distributions (e.g., Beta priors with user-defined parameters) instead of the default uninformative prior.
* **Automated Convergence Diagnostics**: Integrate automated diagnostics like the Geweke or Gelman-Rubin tests to automatically detect when the MCMC chain has converged, rather than relying on a fixed burn-in period.
* **Model Expansion**: Extend the model to handle more complex scenarios, such as A/B/n testing or including covariates to analyze segmented data.

## Reference

Chib, Siddhartha, and Edward Greenberg.  
“Understanding the Metropolis-Hastings Algorithm.”  
_The American Statistician_, vol. 49, no. 4, 1995, pp. 327–335.  
[https://doi.org/10.1080/00031305.1995.10476177](https://doi.org/10.1080/00031305.1995.10476177)

---

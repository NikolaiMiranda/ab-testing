"""
Frequentist vs Bayesian A/B Testing

This script demonstrates both frequentist and Bayesian approaches to A/B testing,
with a focus on learning how MCMC simulation works by implementing Metropolis-Hastings from scratch.

COMPARISON OF APPROACHES:
===========================================

FREQUENTIST APPROACH:
- Classical statistical testing using Z-test for proportions
- Tests H₀: rate_B - rate_A ≤ 0 vs H₁: rate_B - rate_A > 0
- Provides p-value and binary accept/reject decision
- Assumes fixed parameters, focuses on data under null hypothesis
- Interpretation: "If H₀ is true, what's the probability of seeing this data?"

BAYESIAN APPROACH:
- Uses MCMC (Metropolis-Hastings) to sample from posterior distribution
- Computes P(θ_A, θ_B | data) ∝ P(data | θ_A, θ_B) × P(θ_A, θ_B)
- Provides posterior means, credible intervals, and direct probabilities
- Treats parameters as random variables, focuses on parameter uncertainty
- Interpretation: "Given this data, what's the probability θ_B > θ_A?"

SCRIPT STRUCTURE AND FLOW:
===========================================

1. DATA GENERATION (generate_ab_test_data)
   - Simulate A/B test data using true conversion rates
   - Use binomial distribution to model conversions
   - Creates realistic test data for both approaches

2. FREQUENTIST APPROACH (frequentist_ab_test)
   - Perform traditional Z-test for proportions
   - Test H₀: rate_B - rate_A ≤ 0 vs H₁: rate_B - rate_A > 0
   - Provides p-value and accept/reject decision
   - Shows classical statistical testing methodology

3. BAYESIAN MCMC APPROACH (metropolis_hastings + analyze_results)
   - Use Metropolis-Hastings to sample from posterior distribution
   - Construct Markov Chain that converges to P(θ_A, θ_B | data)
   - Compute Bayesian summary statistics from samples
   - Shows modern Bayesian inference with MCMC

4. RESULTS COMPARISON AND VISUALIZATION
   - Compare frequentist vs Bayesian results side-by-side
   - Visualize MCMC convergence and posterior distributions
   - Assess sampling quality and make decisions
   - Demonstrates strengths and limitations of each approach

KEY CONCEPTS:
1. FREQUENTIST: Hypothesis testing, p-values, confidence intervals
2. BAYESIAN: Posterior distributions, credible intervals, direct probabilities
3. MCMC: Use Markov Chain to sample from complex posterior distributions
4. METROPOLIS-HASTINGS: Specific MCMC algorithm that uses acceptance/rejection
5. A/B TESTING: Compare two groups to see which performs better

ALGORITHM OVERVIEW:
- Target: Sample from P(θ_A, θ_B | data) 
- Method: Construct Markov Chain that converges to target distribution
- Proposal: Random walk with Normal(θ_current, σ)
- Acceptance: α = min(1, π(θ_proposed) / π(θ_current))
- Result: Samples that approximate the posterior distribution

This implementation follows the theoretical framework described in:
Chib & Greenberg (1995) "Understanding the Metropolis-Hastings Algorithm"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom, norm
from statsmodels.stats.proportion import proportions_ztest # For frequentist Z-test


def generate_ab_test_data(true_rate_A, true_rate_B, n_A, n_B, seed):
    """
    Generate synthetic A/B test data.

    Parameters:
    - true_rate_A, true_rate_B: True conversion rates
    - n_A, n_B: Number of visitors for each group
    - seed: Random seed for reproducibility

    Returns:
    - conversions_A, conversions_B: Number of conversions for each group
    - n_A, n_B: Number of visitors for each group
    """

    np.random.seed(seed)
    # Use binomial distribution to generate conversions since Binomial is the distribution of the number of successes in a sequence of independent yes/no experiments.
    # In this case, the yes/no experiments are whether a visitor converts or not.
    conversions_A = np.random.binomial(n_A, true_rate_A)
    conversions_B = np.random.binomial(n_B, true_rate_B)

    print("\n--- Simulated A/B Test Data ---")
    print(f"True Conversion Rate for Group A: {true_rate_A:.4f}")
    print(f"True Conversion Rate for Group B: {true_rate_B:.4f}")
    print(f"Group A: {conversions_A} conversions out of {n_A} visitors (Observed rate: {conversions_A/n_A:.4f})")
    print(f"Group B: {conversions_B} conversions out of {n_B} visitors (Observed rate: {conversions_B/n_B:.4f})")

    return conversions_A, n_A, conversions_B, n_B


def frequentist_ab_test(conversions_A, n_A, conversions_B, n_B, alpha=0.05):
    """
    Performs a frequentist A/B test using a Z-test for proportions.
    
    Null Hypothesis (H0): rate_B - rate_A ≤ 0 (Group B is not better than Group A)
    Alternative Hypothesis (H1): rate_B - rate_A > 0 (Group B is better than Group A)

    Parameters:
    - conversions_A, n_A: Conversions and total visitors for group A.
    - conversions_B, n_B: Conversions and total visitors for group B.

    Returns:
    - A dictionary with test decision and basic results.
    """
    print("\n--- Frequentist A/B Test ---")
    print("H0: rate_B - rate_A ≤ 0 (Group B is not better than Group A)")
    print("H1: rate_B - rate_A > 0 (Group B is better than Group A)")
    
    conversion_counts = np.array([conversions_B, conversions_A])
    number_obs = np.array([n_B, n_A])

    # Perform Z-test for one-sided alternative: rate_B - rate_A > 0
    z_stat, p_value = proportions_ztest(conversion_counts, number_obs, alternative='larger')

    # Calculate observed rates and difference
    rate_A = conversions_A / n_A
    rate_B = conversions_B / n_B
    diff_observed = rate_B - rate_A

    # Decision rule: Reject H0 if p-value < α
    decision = "REJECT" if p_value < alpha else "FAIL TO REJECT"
    
    print(f"Observed rates: A = {rate_A:.4f}, B = {rate_B:.4f}")
    print(f"Observed difference (B - A): {diff_observed:.4f}")
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Decision: {decision} the null hypothesis (α = {alpha})")
    
    if p_value < alpha:
        print("Conclusion: Reject H₀ - Group B has a statistically significantly higher conversion rate than Group A")
    else:
        print("Conclusion: Fail to reject H₀ - No statistically significant evidence that Group B is better than Group A")

    frequentist_results = {
        'rate_A': rate_A,
        'rate_B': rate_B,
        'diff_observed': diff_observed,
        'z_stat': z_stat,
        'p_value': p_value,
        'decision': decision
    }
    return frequentist_results


def posterior_function(params, conversions_A, n_A, conversions_B, n_B):
    """
    Calculates the posterior probability for the conversion rates.
    This is the TARGET DISTRIBUTION π(x) that the Metropolis-Hastings algorithm will sample from.
    
    In Bayesian inference: Posterior ∝ Likelihood × Prior
    This function computes P(θ_A, θ_B | data) ∝ P(data | θ_A, θ_B) × P(θ_A, θ_B)
    
    params is a list or array: [theta_A, theta_B]
    """
    # theta is our estimate for the true conversion rate for each group
    theta_A, theta_B = params

    # Return 0 if the parameters are not valid probabilities (0 < theta < 1)
    # This ensures we only sample from valid parameter space
    if not (0 < theta_A < 1 and 0 < theta_B < 1):
        return 0.0

    # PRIORS: P(θ_A) and P(θ_B)
    # Using Beta(1, 1) = Uniform(0, 1) as non-informative prior
    # This means we have no prior knowledge about the conversion rates
    alpha_prior, beta_prior = 1, 1
    prior_A = beta.pdf(theta_A, a=alpha_prior, b=beta_prior)
    prior_B = beta.pdf(theta_B, a=alpha_prior, b=beta_prior)

    # LIKELIHOODS: P(data | θ_A) and P(data | θ_B)
    # Binomial distribution models the number of successes (conversions) 
    # in n independent trials (visitors) with success probability θ
    likelihood_A = binom.pmf(conversions_A, n_A, theta_A)
    likelihood_B = binom.pmf(conversions_B, n_B, theta_B)

    # The Metropolis-Hastings algorithm requires a value proportional to the posterior distribution.
    # Our goal is to sample from the POSTERIOR:
    # P(θ_A, θ_B | data) ∝ P(data | θ_A, θ_B) × P(θ_A, θ_B)
    # Using Bayes' Theorem we get P(θ_A, θ_B | data) = [P(data | θ_A, θ_B) * P(θ_A, θ_B)] / P(data)
    # We can ignore the normalization by P(data) since it will cancel out in the acceptance ratio.
    return likelihood_A * likelihood_B * prior_A * prior_B

def alpha_function(current_params, proposed_params, conversions_A, n_A, conversions_B, n_B):
    """
    Calculate the acceptance probability α for Metropolis-Hastings.
    
    This implements the core M-H acceptance ratio:
    α = min(1, π(y) * q(x|y) / π(x) * q(y|x))
    
    Where:
    - π(x) = target distribution (our posterior)
    - q(y|x) = proposal distribution (probability of observing y given current x)
    - q(x|y) = reverse proposal distribution

    The acceptance ratio is constructed to ensure reversibility which in turn guarantees that 
    the target distribution has an invariant distribution and under mild regularity conditions (irreducibility and aperiodicity) 
    the Markov Chain will converge to it.
    
    Since we're using symmetric proposal distributions (normal distribution),
    q(y|x) = q(x|y), so the proposal densities cancel out:
    α = min(1.0, π(y) / π(x)) = min(1.0, proposed_posterior / current_posterior)
    """
    proposed_posterior = posterior_function(proposed_params, conversions_A, n_A, conversions_B, n_B)
    current_posterior = posterior_function(current_params, conversions_A, n_A, conversions_B, n_B)

    # Avoid division by zero
    if current_posterior == 0:
        return 0.0 
        # If current posterior is 0, we're in a very low probability region
        # This could happen if we're outside the valid parameter space
        # Returning 0 means we'll reject this proposal and stay at current position
    
    # Calculate acceptance ratio: α = min(1, π(y) / π(x))
    # This is the heart of the Metropolis-Hastings algorithm
    # - π(y) = proposed_posterior (unnormalized)
    # - π(x) = current_posterior (unnormalized)
    # - The ratio π(y)/π(x) is the same whether we use normalized or unnormalized posteriors since the P(data) terms cancel out!
    return min(1.0, proposed_posterior / current_posterior)

def metropolis_hastings(conversions_A, n_A, conversions_B, n_B,
                       iterations, proposal_std, initial_params, burn_in_frac=0.2):
    """
    Run Metropolis-Hastings MCMC sampler.
    
    This implements the core Metropolis-Hastings algorithm:
    1. Initialize Markov Chain with X[1]
    2. For each iteration i:
       - Generate proposal Y from q(Y | X[i])
       - Sample U ~ Uniform(0,1)
       - Compute acceptance ratio α = min(1, π(Y)q(X[i]|Y) / π(X[i])q(Y|X[i]))
       - If U ≤ α, accept: X[i+1] ← Y, else reject: X[i+1] ← X[i]
    3. After convergence, samples approximate the target distribution π(x)

    Parameters:
    - conversions_A, conversions_B: Observed conversion counts
    - n_A, n_B: Total visitor counts
    - iterations: Number of MCMC iterations (chain length)
    - burn_in_frac: Fraction of samples to discard as burn-in 
    - proposal_std: Standard deviation for proposal distribution q(Y|X)
    - initial_params: Initial parameter values [theta_A, theta_B]

    Returns:
    - posterior_samples: Post-burn-in samples (converged chain)
    - full_samples: All samples (including burn-in)
    - acceptance_rate: Overall acceptance rate
    - post_burnin_acceptance_rate: Post-burn-in acceptance rate
    - burn_in: Number of burn-in samples
    """
    print(f"Running Metropolis-Hastings with {iterations} iterations...")

    # Calculate burn-in period
    # Burn-in allows the chain to "forget" its starting point and consider the part of the chain post convergence
    burn_in = int(burn_in_frac * iterations)

    # Initialize the Markov Chain
    # We'll track both the samples and whether each step was accepted
    accepted = np.zeros(iterations, dtype=bool)
    samples = np.zeros((iterations, 2))  # 2 parameters: theta_A, theta_B
    samples[0] = initial_params  # X[0] in the algorithm

    # METROPOLIS-HASTINGS ALGORITHM
    # This is the main loop that constructs the Markov Chain
    for t in range(iterations - 1):
        # STEP 1: Generate proposal Y from q(Y | X[t])
        # Using symmetric normal proposal: q(Y|X) = Normal(X, proposal_std)
        # This is a "random walk" proposal - we add noise to current position
        proposed_params = np.random.normal(samples[t], proposal_std)

        # STEP 2: Calculate acceptance probability α
        # α = min(1, π(Y)q(X|Y) / π(X)q(Y|X))
        # Since q is symmetric, this simplifies to α = min(1, π(Y) / π(X))
        acceptance_prob = alpha_function(samples[t], proposed_params, conversions_A, n_A, conversions_B, n_B)

        # STEP 3: Accept or reject using Uniform(0,1) random number
        # This implements the probabilistic acceptance rule
        U = np.random.uniform(0, 1)
        if U < acceptance_prob:
            # ACCEPT: Move to proposed position
            samples[t + 1] = proposed_params
            accepted[t + 1] = True
        else:
            # REJECT: Stay at current position
            samples[t + 1] = samples[t]

    # POST-PROCESSING: Remove burn-in period
    # Only use samples after the chain has converged to the target distribution
    posterior_samples = samples[burn_in:]
    posterior_accepted = accepted[burn_in:]

    # Calculate acceptance rates for diagnostics
    # Ideal acceptance rate is typically between 23-50% for efficient exploration
    acceptance_rate = np.mean(accepted)

    print(f"Posterior samples: {len(posterior_samples)}")
    print(f"Overall acceptance rate: {acceptance_rate:.2%}")

    return posterior_samples, samples, acceptance_rate, burn_in



def analyze_results(posterior_samples, true_rate_A, true_rate_B):
    """
    Analyze MCMC results and calculate summary statistics.
    
    This function takes the samples from our Markov Chain and computes Bayesian
    summary statistics. The key insight is that these samples approximate the
    posterior distribution P(θ_A, θ_B | data), so we can compute any quantity
    of interest by taking expectations over these samples.

    Parameters:
    - posterior_samples: Post-burn-in MCMC samples (approximate posterior)
    - true_rate_A, true_rate_B: True conversion rates (for simulated data comparison)

    Returns:
    - Dictionary containing all analysis results
    """
    # Extract samples for each parameter
    # These samples approximate the marginal posterior distributions
    posterior_A = posterior_samples[:, 0]  # P(θ_A | data)
    posterior_B = posterior_samples[:, 1]  # P(θ_B | data)

    # BAYESIAN SUMMARY STATISTICS
    # These are computed as expectations over the posterior samples
    
    # Posterior means: E[θ_A | data] and E[θ_B | data]
    # These are our "best guess" for the true conversion rates
    posterior_mean_A = np.mean(posterior_A)
    posterior_mean_B = np.mean(posterior_B)
    
    # Probability that B is better than A: P(θ_B > θ_A | data)
    # This is computed by counting what fraction of samples have θ_B > θ_A
    prob_B_better_A = np.mean(posterior_B > posterior_A)
    
    # Expected difference: E[θ_B - θ_A | data]
    # This tells us the expected improvement from using B over A
    expected_difference = np.mean(posterior_B - posterior_A)

    # CREDIBLE INTERVALS (Bayesian confidence intervals)
    # These contain 95% of the posterior probability mass
    # Different from frequentist confidence intervals!
    ci_A = np.quantile(posterior_A, [0.025, 0.975])  # 95% CI for θ_A
    ci_B = np.quantile(posterior_B, [0.025, 0.975])  # 95% CI for θ_B

    # SAMPLING QUALITY DIAGNOSTICS (for simulated data only)
    # These help us assess if our MCMC sampler is working correctly
    
    # Bias: How far off are our posterior means from the true values?
    # Should be close to 0 if sampler is working well
    bias_A = posterior_mean_A - true_rate_A
    bias_B = posterior_mean_B - true_rate_B

    # Coverage: Do our credible intervals contain the true values?
    # Should be true ~95% of the time for 95% CIs
    a_in_ci = ci_A[0] <= true_rate_A <= ci_A[1]
    b_in_ci = ci_B[0] <= true_rate_B <= ci_B[1]

    # Print Bayesian analysis results
    print("\n--- Bayesian A/B Test Results ---")
    print(f"Posterior Mean for Group A: {posterior_mean_A:.4f}")
    print(f"95% Credible Interval for Group A: ({ci_A[0]:.4f}, {ci_A[1]:.4f})")
    print(f"Posterior Mean for Group B: {posterior_mean_B:.4f}")
    print(f"95% Credible Interval for Group B: ({ci_B[0]:.4f}, {ci_B[1]:.4f})")
    print(f"Probability that Group B is better than Group A: {prob_B_better_A:.2%}")
    print(f"Expected difference (B - A) in conversion rates: {expected_difference:.4f}")

    print(f"\n--- MCMC Sampling Quality Diagnostics ---")
    print(f"Bias for A: {bias_A:.4f}")
    print(f"Bias for B: {bias_B:.4f}")
    print(f"True A ({true_rate_A:.4f}) in 95% CI: {a_in_ci}")
    print(f"True B ({true_rate_B:.4f}) in 95% CI: {b_in_ci}")

    print("\n--- Bayesian Decision Making ---")
    # Bayesian decision making uses posterior probabilities directly
    print(f"There's a {prob_B_better_A:.2%} probability that Group B has a higher true conversion rate than Group A.")    

    results = {
        'posterior_A': posterior_A,
        'posterior_B': posterior_B,
        'posterior_mean_A': posterior_mean_A,
        'posterior_mean_B': posterior_mean_B,
        'prob_B_better_A': prob_B_better_A,
        'expected_difference': expected_difference,
        'ci_A': ci_A,
        'ci_B': ci_B,
        'bias_A': bias_A,
        'bias_B': bias_B,
        'a_in_ci': a_in_ci,
        'b_in_ci': b_in_ci
    }

    return results

def plot_results(samples, posterior_samples, burn_in, true_rate_A, true_rate_B, results):
    """
    Create comprehensive visualization of MCMC results.

    Parameters:
    - samples: All MCMC samples (including burn-in)
    - posterior_samples: Post-burn-in samples
    - burn_in: Number of burn-in samples
    - true_rate_A, true_rate_B: True conversion rates (for plotting simulated data)
    - results: Analysis results dictionary
    """
    plt.figure(figsize=(18, 6)) # Increased figure size for better visibility

    # Trace plot to check convergence (show full trace with burn-in marked)
    plt.subplot(1, 3, 1)
    plt.plot(samples[:, 0], alpha=0.6, label='theta_A')
    plt.plot(samples[:, 1], alpha=0.6, label='theta_B')
    plt.axvline(burn_in, color='red', linestyle='--', alpha=0.7, label='Burn-in')
    plt.title('MCMC Trace Plot (with Burn-in)')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    # Posterior distributions (using post-burn-in samples)
    plt.subplot(1, 3, 2)
    plt.hist(results['posterior_A'], bins=50, density=True, alpha=0.6, label='Group A Posterior', color='skyblue')
    plt.hist(results['posterior_B'], bins=50, density=True, alpha=0.6, label='Group B Posterior', color='lightcoral')
    plt.axvline(np.mean(results['posterior_A']), color='blue', linestyle='--', label=f'Mean A: {np.mean(results["posterior_A"]):.4f}')
    plt.axvline(np.mean(results['posterior_B']), color='red', linestyle='--', label=f'Mean B: {np.mean(results["posterior_B"]):.4f}')
    plt.title('Posterior Distributions (Post-Burn-in)')
    plt.xlabel('Conversion Rate')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    # Scatter plot of joint posterior (using post-burn-in samples)
    plt.subplot(1, 3, 3)
    plt.scatter(results['posterior_A'], results['posterior_B'], alpha=0.1, s=1)
    plt.plot([true_rate_A], [true_rate_B], 'ro', markersize=8, label='True Values')
    plt.title('Joint Posterior (Post-Burn-in)')
    plt.xlabel('theta_A')
    plt.ylabel('theta_B')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------

def main():
    """
    Main execution function demonstrating both frequentist and Bayesian A/B testing.
    
    This function implements a complete comparison between two statistical paradigms:
    1. Generate synthetic data (same for both approaches)
    2. Apply frequentist approach (Z-test with p-values)
    3. Apply Bayesian approach (MCMC with posterior probabilities)
    4. Compare and visualize results from both approaches
    
    This demonstrates how the same data can be analyzed using different statistical
    philosophies, highlighting the strengths and limitations of each approach.
    """

    # Define true rates and sample sizes for simulation
    seed = 3
    alpha = 0.05
    true_rate_A = 0.10
    true_rate_B = 0.1075 # Group B is slightly better
    n_A = 1000
    n_B = 1000
    iterations = 50000 # mcmc simulation iterations
    proposal_std = [0.015, 0.015] # use this to tweak acceptance rate

    # STEP 1: DATA GENERATION
    # Create synthetic A/B test data to work with
    print("Step 1: Generating synthetic A/B test data...")
    conversions_A, n_A, conversions_B, n_B = generate_ab_test_data(true_rate_A, true_rate_B, n_A, n_B, seed)

    # STEP 2: FREQUENTIST APPROACH
    # Apply traditional statistical testing
    frequentist_results = frequentist_ab_test(conversions_A, n_A, conversions_B, n_B, alpha)

    # STEP 3: BAYESIAN MCMC APPROACH
    # Use Metropolis-Hastings to sample from posterior distribution
    print("\nStep 3: Running Metropolis-Hastings MCMC...")
    # Initial parameters can be set closer to observed rates for faster convergence if desired
    initial_A_rate = conversions_A / n_A 
    initial_B_rate = conversions_B / n_B 
    posterior_samples, full_samples, acceptance_rate, burn_in = metropolis_hastings(
        conversions_A, n_A, conversions_B, n_B, iterations, proposal_std,
        initial_params=[initial_A_rate, initial_B_rate]
    )

    # STEP 4: BAYESIAN ANALYSIS
    # Compute summary statistics from MCMC samples
    print("\nStep 4: Analyzing Bayesian results...")
    bayesian_results = analyze_results(posterior_samples, true_rate_A, true_rate_B)

    # STEP 5: VISUALIZATION
    # Create plots to visualize MCMC convergence and posterior distributions
    print("\nStep 6: Creating visualizations...")
    plot_results(full_samples, posterior_samples, burn_in, true_rate_A, true_rate_B, bayesian_results)

if __name__ == "__main__":
    main()
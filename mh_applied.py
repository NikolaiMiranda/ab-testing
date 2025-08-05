import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom, norm

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def generate_ab_test_data(true_rate_A=0.10, true_rate_B=0.12, n_A=5000, n_B=5000, seed=3):
    """
    Generate synthetic A/B test data.
    
    Parameters:
    - true_rate_A, true_rate_B: True conversion rates
    - n_A, n_B: Number of visitors for each group
    - seed: Random seed for reproducibility
    
    Returns:
    - conversions_A, conversions_B: Number of conversions for each group
    """
    np.random.seed(seed)
    conversions_A = np.random.binomial(n_A, true_rate_A)
    conversions_B = np.random.binomial(n_B, true_rate_B)
    
    print(f"Group A: {conversions_A} conversions out of {n_A} visitors.")
    print(f"Group B: {conversions_B} conversions out of {n_B} visitors.")
    
    return conversions_A, conversions_B

def posterior_function(params, conversions_A, n_A, conversions_B, n_B):
    """
    Calculates the posterior probability for the conversion rates.
    This is the function the Metropolis-Hastings algorithm will sample from.
    params is a list or array: [theta_A, theta_B]
    """
    theta_A, theta_B = params
    
    # Return 0 if the parameters are not valid probabilities (0 < theta < 1)
    if not (0 < theta_A < 1 and 0 < theta_B < 1):
        return 0.0

    # Priors (using a non-informative Beta(1, 1) which is a uniform prior)
    alpha_prior, beta_prior = 1, 1
    prior_A = beta.pdf(theta_A, a=alpha_prior, b=beta_prior)
    prior_B = beta.pdf(theta_B, a=alpha_prior, b=beta_prior)

    # Likelihoods (Binomial distribution for conversion data)
    likelihood_A = binom.pmf(conversions_A, n_A, theta_A)
    likelihood_B = binom.pmf(conversions_B, n_B, theta_B)
    
    return likelihood_A * likelihood_B * prior_A * prior_B

def alpha_function(current_params, proposed_params, conversions_A, n_A, conversions_B, n_B):
    """
    Calculate the acceptance probability for Metropolis-Hastings.
    
    Full Metropolis-Hastings acceptance ratio:
    α = min(1, π(y) * q(x|y) / π(x) * q(y|x))
    
    Where:
    - π(x) = target distribution (our posterior)
    - q(y|x) = proposal distribution (probability of proposing y given current x)
    - q(x|y) = reverse proposal distribution
    
    Since we're using symmetric proposal distributions (normal distribution),
    q(y|x) = q(x|y), so the proposal densities cancel out:
    α = min(1, π(y) / π(x))
    """
    
    current_posterior = posterior_function(current_params, conversions_A, n_A, conversions_B, n_B)
    proposed_posterior = posterior_function(proposed_params, conversions_A, n_A, conversions_B, n_B)
    
    # Avoid division by zero
    if current_posterior == 0:
        return 0.0
    
    return min(1.0, proposed_posterior / current_posterior)

def metropolis_hastings(conversions_A, n_A, conversions_B, n_B, 
                       Nmcmc=10000, burn_in_frac=0.2, 
                       proposal_std=[0.01, 0.01], initial_params=[0.1, 0.1]):
    """
    Run Metropolis-Hastings MCMC sampler for A/B test.
    
    Parameters:
    - conversions_A, conversions_B: Observed conversion counts
    - n_A, n_B: Total visitor counts
    - Nmcmc: Number of MCMC iterations
    - burn_in_frac: Fraction of samples to discard as burn-in
    - proposal_std: Standard deviation for proposal distribution
    - initial_params: Initial parameter values [theta_A, theta_B]
    
    Returns:
    - posterior_samples: Post-burn-in samples
    - full_samples: All samples (including burn-in)
    - acceptance_rate: Overall acceptance rate
    - post_burnin_acceptance_rate: Post-burn-in acceptance rate
    - burn_in: Number of burn-in samples
    """
    print(f"Running Metropolis-Hastings with {Nmcmc} iterations...")
    
    # Calculate burn-in
    burn_in = int(burn_in_frac * Nmcmc)
    
    # Initialize arrays
    accepted = np.zeros(Nmcmc, dtype=bool)
    samples = np.zeros((Nmcmc, 2))
    samples[0] = initial_params
    
    # Metropolis-Hastings Algorithm
    for t in range(Nmcmc - 1):
        # Propose new parameters
        proposed_params = np.random.normal(samples[t], proposal_std)
        
        # Calculate acceptance probability
        acceptance_prob = alpha_function(samples[t], proposed_params, 
                                       conversions_A, n_A, conversions_B, n_B)
        
        # Accept or reject
        U = np.random.uniform(0, 1)
        if U < acceptance_prob:
            samples[t + 1] = proposed_params
            accepted[t + 1] = True
        else:
            samples[t + 1] = samples[t]
    
    # Discard burn-in samples
    posterior_samples = samples[burn_in:]
    posterior_accepted = accepted[burn_in:]
    
    # Calculate acceptance rates
    acceptance_rate = np.mean(accepted)
    post_burnin_acceptance_rate = np.mean(posterior_accepted)
    
    print(f"Burn-in discarded: {burn_in}")
    print(f"Posterior samples: {len(posterior_samples)}")
    print(f"Overall acceptance rate: {acceptance_rate:.2%}")
    print(f"Post-burn-in acceptance rate: {post_burnin_acceptance_rate:.2%}")
    
    return posterior_samples, samples, acceptance_rate, post_burnin_acceptance_rate, burn_in

def analyze_results(posterior_samples, true_rate_A, true_rate_B):
    """
    Analyze MCMC results and calculate summary statistics.
    
    Parameters:
    - posterior_samples: Post-burn-in MCMC samples
    - true_rate_A, true_rate_B: True conversion rates
    
    Returns:
    - Dictionary containing all analysis results
    """
    posterior_A = posterior_samples[:, 0]
    posterior_B = posterior_samples[:, 1]
    
    # Calculate summary statistics
    posterior_mean_A = np.mean(posterior_A)
    posterior_mean_B = np.mean(posterior_B)
    prob_B_better_A = np.mean(posterior_B > posterior_A)
    
    # Calculate credible intervals
    ci_A = np.quantile(posterior_A, [0.025, 0.975])
    ci_B = np.quantile(posterior_B, [0.025, 0.975])
    
    # Calculate bias
    bias_A = posterior_mean_A - true_rate_A
    bias_B = posterior_mean_B - true_rate_B
    
    # Check if true values are within credible intervals
    a_in_ci = ci_A[0] <= true_rate_A <= ci_A[1]
    b_in_ci = ci_B[0] <= true_rate_B <= ci_B[1]
    
    results = {
        'posterior_A': posterior_A,
        'posterior_B': posterior_B,
        'posterior_mean_A': posterior_mean_A,
        'posterior_mean_B': posterior_mean_B,
        'prob_B_better_A': prob_B_better_A,
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
    - true_rate_A, true_rate_B: True conversion rates
    - results: Analysis results dictionary
    """
    plt.figure(figsize=(15, 5))
    
    # Trace plot to check convergence (show full trace with burn-in marked)
    plt.subplot(1, 3, 1)
    plt.plot(samples[:, 0], alpha=0.6, label='theta_A')
    plt.plot(samples[:, 1], alpha=0.6, label='theta_B')
    plt.axvline(burn_in, color='red', linestyle='--', alpha=0.7, label='Burn-in')
    plt.axhline(true_rate_A, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(true_rate_B, color='orange', linestyle='--', alpha=0.7)
    plt.title('MCMC Trace Plot (with Burn-in)')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.legend()
    
    # Posterior distributions (using post-burn-in samples)
    plt.subplot(1, 3, 2)
    plt.hist(results['posterior_A'], bins=50, density=True, alpha=0.6, label='Group A Posterior')
    plt.hist(results['posterior_B'], bins=50, density=True, alpha=0.6, label='Group B Posterior')
    plt.axvline(true_rate_A, color='blue', linestyle='--', label='True Rate A')
    plt.axvline(true_rate_B, color='orange', linestyle='--', label='True Rate B')
    plt.title('Posterior Distributions (Post-Burn-in)')
    plt.xlabel('Conversion Rate')
    plt.ylabel('Density')
    plt.legend()
    
    # Scatter plot of joint posterior (using post-burn-in samples)
    plt.subplot(1, 3, 3)
    plt.scatter(results['posterior_A'], results['posterior_B'], alpha=0.1, s=1)
    plt.axvline(true_rate_A, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(true_rate_B, color='orange', linestyle='--', alpha=0.7)
    plt.plot([true_rate_A], [true_rate_B], 'ro', markersize=8, label='True Values')
    plt.title('Joint Posterior (Post-Burn-in)')
    plt.xlabel('theta_A')
    plt.ylabel('theta_B')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_summary(results, true_rate_A, true_rate_B):
    """
    Print comprehensive summary of results.
    
    Parameters:
    - results: Analysis results dictionary
    - true_rate_A, true_rate_B: True conversion rates
    """
    print("\n--- Summary of Results (Post-Burn-in) ---")
    print(f"Posterior Mean for Group A: {results['posterior_mean_A']:.4f}")
    print(f"95% Credible Interval for Group A: ({results['ci_A'][0]:.4f}, {results['ci_A'][1]:.4f})")
    print(f"\nPosterior Mean for Group B: {results['posterior_mean_B']:.4f}")
    print(f"95% Credible Interval for Group B: ({results['ci_B'][0]:.4f}, {results['ci_B'][1]:.4f})")
    print(f"\nProbability that Group B is better than Group A: {results['prob_B_better_A']:.2%}")
    
    print(f"\n--- Sampling Quality ---")
    print(f"True rates: A={true_rate_A:.4f}, B={true_rate_B:.4f}")
    print(f"Posterior means: A={results['posterior_mean_A']:.4f}, B={results['posterior_mean_B']:.4f}")
    print(f"Bias: A={results['bias_A']:.4f}, B={results['bias_B']:.4f}")
    print(f"True A in 95% CI: {results['a_in_ci']}")
    print(f"True B in 95% CI: {results['b_in_ci']}")

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------

def main():
    """Main execution function - now very clean and simple!"""
    
    # Step 1: Generate data
    print("Step 1: Generating synthetic A/B test data...")
    conversions_A, conversions_B = generate_ab_test_data()
    
    # Step 2: Run MCMC
    print("\nStep 2: Running Metropolis-Hastings MCMC...")
    posterior_samples, full_samples, acceptance_rate, post_burnin_acceptance_rate, burn_in = metropolis_hastings(
        conversions_A, 5000, conversions_B, 5000
    )
    
    # Step 3: Analyze results
    print("\nStep 3: Analyzing results...")
    results = analyze_results(posterior_samples, 0.10, 0.12) # edit this so its not hardcoded
    
    # Step 4: Print summary
    print_summary(results, 0.10, 0.12)
    
    # Step 5: Visualize results
    print("\nStep 4: Creating visualizations...")
    plot_results(full_samples, posterior_samples, burn_in, 0.10, 0.12, results) # edit this so its not hardcoded

if __name__ == "__main__":
    main()
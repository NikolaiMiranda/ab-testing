import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom, norm

# ----------------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC DATA FOR A/B TEST
# ----------------------------------------------------------------------
print("Step 1: Generating synthetic A/B test data...")
np.random.seed(3)

# True conversion rates for our two versions
true_rate_A = 0.10
true_rate_B = 0.12

# Number of visitors for each group
n_A = 5000
n_B = 5000

# Simulate the number of conversions for each group
conversions_A = np.random.binomial(n_A, true_rate_A)
conversions_B = np.random.binomial(n_B, true_rate_B)

print(f"Group A: {conversions_A} conversions out of {n_A} visitors.")
print(f"Group B: {conversions_B} conversions out of {n_B} visitors.")

# ----------------------------------------------------------------------
# STEP 2: DEFINE THE LOG-POSTERIOR FUNCTION
# ----------------------------------------------------------------------
print("\nStep 2: Defining the log-posterior function...")

def log_posterior(params, conversions_A, n_A, conversions_B, n_B):
    """
    Calculates the log-posterior probability for the conversion rates.
    This is the function the Metropolis-Hastings algorithm will sample from.
    params is a list or array: [theta_A, theta_B]
    """
    theta_A, theta_B = params
    
    # Return -inf if the parameters are not valid probabilities (0 < theta < 1)
    if not (0 < theta_A < 1 and 0 < theta_B < 1):
        return -np.inf

    # Log-Priors (using a non-informative Beta(1, 1) which is a uniform prior)
    alpha_prior, beta_prior = 1, 1
    log_prior_A = beta.logpdf(theta_A, a=alpha_prior, b=beta_prior)
    log_prior_B = beta.logpdf(theta_B, a=alpha_prior, b=beta_prior)

    # Log-Likelihoods (Binomial distribution for conversion data)
    log_likelihood_A = binom.logpmf(conversions_A, n_A, theta_A)
    log_likelihood_B = binom.logpmf(conversions_B, n_B, theta_B)
    
    return log_likelihood_A + log_likelihood_B + log_prior_A + log_prior_B

# ----------------------------------------------------------------------
# STEP 3: IMPLEMENT THE METROPOLIS-HASTINGS ALGORITHM
# ----------------------------------------------------------------------
print("\nStep 3: Running the Metropolis-Hastings algorithm...")

def metropolis_hastings(n_iterations, initial_params, proposal_std, conversions_A, n_A, conversions_B, n_B):
    """
    Metropolis-Hastings MCMC sampler for the A/B test.
    """
    n_params = len(initial_params)
    samples = np.zeros((n_iterations, n_params))
    current_params = np.array(initial_params)
    accepted = 0  # Track number of accepted proposals

    # The MCMC loop
    for i in range(n_iterations):
        # Propose new parameters for theta_A and theta_B
        proposed_params = np.random.normal(current_params, proposal_std)

        # Calculate the log-posterior for the current and proposed parameters
        current_log_post = log_posterior(current_params, conversions_A, n_A, conversions_B, n_B)
        proposed_log_post = log_posterior(proposed_params, conversions_A, n_A, conversions_B, n_B)

        # Calculate the acceptance ratio (using log-probabilities)
        alpha = np.exp(proposed_log_post - current_log_post)

        # Accept or reject the new sample
        if np.random.uniform(0, 1) < alpha:
            current_params = proposed_params
            accepted += 1
        
        samples[i] = current_params
    
    acceptance_rate = accepted / n_iterations
    return samples, acceptance_rate

# MCMC settings
n_iterations = 500000  # Increased from 50000
initial_params = [0.1, 0.1]  # Initial guess for [theta_A, theta_B]
proposal_std = [0.0075, 0.0075]  # Tune to adjust acceptance rate

# Run the sampler
mcmc_samples, acceptance_rate = metropolis_hastings(n_iterations, initial_params, proposal_std, conversions_A, n_A, conversions_B, n_B)

# Discard burn-in samples (increased burn-in)
burn_in = 50000  
posterior_samples = mcmc_samples[burn_in:]

# Add convergence diagnostics
print(f"\nMCMC Diagnostics:")
print(f"Total iterations: {n_iterations}")
print(f"Burn-in discarded: {burn_in}")
print(f"Posterior samples: {len(posterior_samples)}")
print(f"Acceptance rate: {acceptance_rate:.2%}")

# Check if samples are exploring the space
print(f"Parameter ranges explored:")
print(f"theta_A: [{np.min(posterior_samples[:, 0]):.4f}, {np.max(posterior_samples[:, 0]):.4f}]")
print(f"theta_B: [{np.min(posterior_samples[:, 1]):.4f}, {np.max(posterior_samples[:, 1]):.4f}]")

# ----------------------------------------------------------------------
# STEP 4: ANALYZE AND VISUALIZE THE RESULTS
# ----------------------------------------------------------------------
print("\nStep 4: Analyzing and visualizing the results...")

# Separate the samples for each group
posterior_A = posterior_samples[:, 0]
posterior_B = posterior_samples[:, 1]

# Plot posterior distributions
plt.figure(figsize=(15, 5))

# Trace plot to check convergence
plt.subplot(1, 3, 1)
plt.plot(mcmc_samples[burn_in:, 0], alpha=0.6, label='theta_A')
plt.plot(mcmc_samples[burn_in:, 1], alpha=0.6, label='theta_B')
plt.axhline(true_rate_A, color='blue', linestyle='--', alpha=0.7)
plt.axhline(true_rate_B, color='orange', linestyle='--', alpha=0.7)
plt.title('MCMC Trace Plot')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.legend()

# Posterior distributions
plt.subplot(1, 3, 2)
plt.hist(posterior_A, bins=50, density=True, alpha=0.6, label='Group A Posterior')
plt.hist(posterior_B, bins=50, density=True, alpha=0.6, label='Group B Posterior')
plt.axvline(true_rate_A, color='blue', linestyle='--', label='True Rate A')
plt.axvline(true_rate_B, color='orange', linestyle='--', label='True Rate B')
plt.title('Posterior Distributions')
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.legend()

# Scatter plot of joint posterior
plt.subplot(1, 3, 3)
plt.scatter(posterior_A, posterior_B, alpha=0.1, s=1)
plt.axvline(true_rate_A, color='blue', linestyle='--', alpha=0.7)
plt.axhline(true_rate_B, color='orange', linestyle='--', alpha=0.7)
plt.plot([true_rate_A], [true_rate_B], 'ro', markersize=8, label='True Values')
plt.title('Joint Posterior')
plt.xlabel('theta_A')
plt.ylabel('theta_B')
plt.legend()

# Calculate the probability that B is better than A
prob_B_better_A = np.mean(posterior_B > posterior_A)

# Calculate credible intervals for each rate
ci_A = np.quantile(posterior_A, [0.025, 0.975])
ci_B = np.quantile(posterior_B, [0.025, 0.975])

# Print final results
print("\n--- Summary of Results ---")
print(f"Posterior Mean for Group A: {np.mean(posterior_A):.4f}")
print(f"95% Credible Interval for Group A: ({ci_A[0]:.4f}, {ci_A[1]:.4f})")
print(f"\nPosterior Mean for Group B: {np.mean(posterior_B):.4f}")
print(f"95% Credible Interval for Group B: ({ci_B[0]:.4f}, {ci_B[1]:.4f})")
print(f"\nProbability that Group B is better than Group A: {prob_B_better_A:.2%}")

# Additional diagnostics
print(f"\n--- Sampling Quality ---")
print(f"True rates: A={true_rate_A:.4f}, B={true_rate_B:.4f}")
print(f"Posterior means: A={np.mean(posterior_A):.4f}, B={np.mean(posterior_B):.4f}")
print(f"Bias: A={np.mean(posterior_A) - true_rate_A:.4f}, B={np.mean(posterior_B) - true_rate_B:.4f}")

# Check if true values are within credible intervals
a_in_ci = ci_A[0] <= true_rate_A <= ci_A[1]
b_in_ci = ci_B[0] <= true_rate_B <= ci_B[1]
print(f"True A in 95% CI: {a_in_ci}")
print(f"True B in 95% CI: {b_in_ci}")

# Show the plot after printing results
plt.show()
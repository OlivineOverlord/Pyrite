import numpy as np
import emcee
import scipy.optimize as opt


def complex_step_derivative(f, x, params, dx=1e-20):
    """Computes df/dx using the complex-step derivative approximation."""
    return np.imag(f(x + 1j * dx, *params)) / dx


def chi2_effective_variance(params, x, y, sigma_x, sigma_y, model_function):
    """Computes chi-square value considering errors in both x and y."""
    y_model = model_function(x, *params)  # Model predictions
    df_dx = complex_step_derivative(model_function, x, params)  # Compute df/dx

    # Handle cases where sigma_x or sigma_y is zero
    if np.any(sigma_x == 0):
        sigma_eff = sigma_y  # Ignore sigma_x contribution
    elif np.any(sigma_y == 0):
        sigma_eff = np.abs(df_dx) * sigma_x  # Ignore sigma_y contribution
    else:
        sigma_eff = np.sqrt(sigma_y**2 + (df_dx**2 * sigma_x**2))  # Normal case
    sigma_eff = np.clip(sigma_eff, 1e-8, np.inf)

    residuals = (y - y_model) / sigma_eff  # Weighted residuals
    return np.sum(residuals**2)  # Chi-square value


def fit_effective_variance(x, y, sigma_x, sigma_y, model_function, initial_guess):
    """Fits any model using the Effective Variance Method with complex-step differentiation."""
    result = opt.minimize(
        chi2_effective_variance, initial_guess, args=(x, y, sigma_x, sigma_y, model_function), method='L-BFGS-B'
    )
    if result.success:
        fit_params = result.x  # Best-fit parameters
        return fit_params
    else:
        raise RuntimeError("Optimization failed: " + result.message)


def compute_r2(x, y, sigma_x, sigma_y, model_function, fit_params):
    """Computes the coefficient of determination (R^2) for the fit."""
    y_model = model_function(x, *fit_params)  # Predicted values
    residuals = y - y_model  # Residuals
    ss_res = np.sum(residuals**2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)  # R^2 value


def compute_weighted_r2(x, y, sigma_x, sigma_y, model_function, fit_params):
    """Computes the weighted R^2 using effective variance weights."""

    y_model = model_function(x, *fit_params)  # Model predictions
    df_dx = complex_step_derivative(model_function, x, fit_params)  # Compute df/dx

    # Handle cases where sigma_x or sigma_y are zero
    if np.any(sigma_x == 0):
        sigma_eff = sigma_y  # Ignore sigma_x contribution
    elif np.any(sigma_y == 0):
        sigma_eff = np.abs(df_dx) * sigma_x  # Ignore sigma_y contribution
    else:
        sigma_eff = np.sqrt(sigma_y**2 + (df_dx**2 * sigma_x**2))  # Normal case

    weights = 1 / sigma_eff**2  # Weight for each point
    weighted_mean_y = np.sum(weights * y) / np.sum(weights)  # Weighted mean of y

    ss_res = np.sum(weights * (y - y_model)**2)  # Weighted residual sum of squares
    ss_tot = np.sum(weights * (y - weighted_mean_y)**2)  # Weighted total sum of squares

    return 1 - (ss_res / ss_tot)  # Weighted R^2

def compute_chi2_red(x, y, sigma_x, sigma_y, model_function, fit_params):
    """Computes chi-square, reduced chi-square, and returns both."""
    chi2 = chi2_effective_variance(fit_params, x, y, sigma_x, sigma_y, model_function)
    dof = len(x) - len(fit_params)  # Degrees of freedom
    chi2_red = chi2 / dof if dof > 0 else np.nan  # Avoid division by zero
    return chi2, chi2_red


def aic_bic_from_fit(fit_params, x, y, sigma_x, sigma_y, model_function):
    """
    Compute AIC, AICc, and BIC given fitted parameters and the full log-likelihood.

    Parameters
    ----------
    fit_params : array
        Best-fit parameters (e.g., from optimizer).
    x, y : arrays
        Data points.
    sigma_x, sigma_y : arrays
        Uncertainties in x and y.
    model_function : callable
        Model function f(x, *params).

    Returns
    -------
    dict with logL, AIC, AICc, BIC
    """
    # Compute full log-likelihood
    y_model = model_function(x, *fit_params)
    df_dx = complex_step_derivative(model_function, x, fit_params)
    sigma_eff2 = sigma_y**2 + (df_dx**2) * sigma_x**2
    sigma_eff2 = np.clip(sigma_eff2, 1e-300, np.inf)  # avoid log(0)
    resid = y - y_model

    logL = -0.5 * np.sum((resid**2) / sigma_eff2 + np.log(2*np.pi*sigma_eff2))

    # Information Criteria
    n = len(x)
    k = len(fit_params)

    AIC = 2 * k - 2 * logL
    AICc = np.inf
    if n - k - 1 > 0:
        AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    BIC = k * np.log(n) - 2 * logL

    return {"logL": logL, "AIC": AIC, "AICc": AICc, "BIC": BIC}


def estimate_uncertainties_mcmc(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    model_function,
    fit_params: np.ndarray,
    n_walkers: int = 50,
    n_steps: int = 5000,
    burn_in: int = 1000,
    prior_bounds = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates parameter uncertainties using MCMC while accounting for non-normality and correlations.

    Parameters:
        x, y: Data points
        sigma_x, sigma_y: Uncertainties in x and y
        model_function: The function describing the model
        fit_params: Best-fit parameters from optimization
        n_walkers: Number of MCMC walkers (default=50)
        n_steps: Number of MCMC steps (default=5000)
        burn_in: Number of steps to discard as burn-in (default=1000)
        prior_bounds: List of (min, max) tuples for each parameter. If None, assumes wide bounds.

    Returns:
        param_samples: A set of posterior samples for each parameter
        param_means: Mean of each parameter
        param_uncertainties: Standard deviation of each parameter
        param_cov_matrix: Covariance matrix of the parameters
    """
    # Initialize walkers around the best-fit parameters with small Gaussian noise
    ndim = len(fit_params)
    if prior_bounds is None:
        prior_bounds = [(-1e6, 1e6)] * ndim

    def log_likelihood(params):
        try:
            y_model = model_function(x, *params)
            df_dx = complex_step_derivative(model_function, x, params)
            sigma_eff2 = sigma_y**2 + (df_dx**2 * sigma_x**2)
            sigma_eff2 = np.clip(sigma_eff2, 1e-300, np.inf)

            resid = y - y_model
            logL = -0.5 * np.sum((resid**2) / sigma_eff2 + np.log(2 * np.pi * sigma_eff2))
            return logL
        except Exception:
            return -np.inf

    def log_prior(params):
        if prior_bounds is None:
            return 1E-6 if np.all(np.isfinite(params)) else -np.inf
        for p, (pmin, pmax) in zip(params, prior_bounds):
            if not np.isfinite(p) or p < pmin or p > pmax:
                return -np.inf
        return 1E-6

    def log_posterior(params):
        """Posterior = Prior + Likelihood (log space)."""
        lp = log_prior(params)
        return lp + log_likelihood(params) if np.isfinite(lp) else -np.inf

    # Initialize walkers
    initial_positions = []
    for _ in range(n_walkers):
        scale = np.maximum(1e-4 * np.abs(fit_params), 1e-6)
        pos = fit_params + scale * np.random.randn(ndim)
        # Clamp to prior bounds
        pos = np.clip(pos, [b[0] for b in prior_bounds], [b[1] for b in prior_bounds])
        initial_positions.append(pos)
    initial_positions = np.array(initial_positions)

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(initial_positions, n_steps, progress=True)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    param_means = np.mean(samples, axis=0)
    param_uncertainties = np.std(samples, axis=0)
    param_cov_matrix = np.cov(samples.T)  # Covariance matrix

    return samples, param_means, param_uncertainties, param_cov_matrix


def fit_effective_variance_full(x, y, sigma_x, sigma_y, model_function, initial_guess, use_mcmc=True, n_walkers=50, n_steps=5000, burn_in=1000, prior_bounds=None):
    """
    Fits a model using the Effective Variance Method, considering uncertainties in both x and y.

    Parameters:
        x, y: Data points
        sigma_x, sigma_y: Uncertainties in x and y
        model_function: The function describing the model
        initial_guess: Initial guess for parameters
        use_mcmc: If True, estimates uncertainties using MCMC (default=True)
        n_walkers: Number of MCMC walkers (default=50)
        n_steps: Number of MCMC steps (default=5000)
        burn_in: Number of steps to discard as burn-in (default=1000)

    Returns:
        results: Dictionary containing fit statistics and diagnostics.
    """
    # Perform optimization using the effective variance chi-square function
    result = opt.minimize(
        chi2_effective_variance, initial_guess, args=(x, y, sigma_x, sigma_y, model_function)
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    fit_params = result.x  # Best-fit parameters

    # Compute predicted values and residuals
    y_model = model_function(x, *fit_params)
    residuals = y - y_model

    # Compute chi² and reduced chi²
    chi2, chi2_red = compute_chi2_red(x, y, sigma_x, sigma_y, model_function, fit_params)

    # Compute weighted R²
    weighted_r2 = compute_weighted_r2(x, y, sigma_x, sigma_y, model_function, fit_params)

    # AIC and BIC
    info_crit = aic_bic_from_fit(fit_params, x, y, sigma_x, sigma_y, model_function)

    # Compute degrees of freedom
    dof = len(x) - len(fit_params)

    # MCMC Uncertainty Estimation
    param_samples, param_means, param_uncertainties_mcmc, param_cov_matrix_mcmc = (None, None, None, None)
    if use_mcmc:
        param_samples, param_means, param_uncertainties_mcmc, param_cov_matrix_mcmc = estimate_uncertainties_mcmc(
            x, y, sigma_x, sigma_y, model_function, fit_params, n_walkers, n_steps, burn_in, prior_bounds
        )

    return {
        "fit_params": fit_params,
        "weighted_r2": weighted_r2,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "aic_c": info_crit["AICc"],
        "bic": info_crit["BIC"],
        "residuals": residuals,
        "dof": dof,
        "y_model": y_model,
        "param_samples_mcmc": param_samples,
        "param_means_mcmc": param_means,
        "param_uncertainties_mcmc": param_uncertainties_mcmc,
        "covariance_matrix_mcmc": param_cov_matrix_mcmc,
    }


def compute_credible_band(x_vals, model_function, param_samples, ci=68, chi2_red=None):
    """
    Compute the prediction mean and credible interval for a model
    using MCMC posterior parameter samples. Optionally scales the
    credible interval by sqrt(reduced chi-square).

    Parameters:
        x_vals (array): x values to evaluate the model at
        model_function (callable): model function f(x, *params)
        param_samples (array): MCMC samples (n_samples x n_params)
        ci (float): Confidence interval percentage (e.g., 68, 95)
        chi2_red (float or None): Reduced chi-square value; if provided,
                                  scales the credible interval by sqrt(chi2_red)

    Returns:
        y_pred_mean (array): Mean model prediction
        y_lower (array): Lower bound of credible interval (possibly scaled)
        y_upper (array): Upper bound of credible interval (possibly scaled)
    """
    n_samples = len(param_samples)
    y_pred_samples = np.zeros((n_samples, len(x_vals)))

    for i, params in enumerate(param_samples):
        y_pred_samples[i] = model_function(x_vals, *params)

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_lower = np.percentile(y_pred_samples, lower_percentile, axis=0)
    y_upper = np.percentile(y_pred_samples, upper_percentile, axis=0)

    if chi2_red is not None:
        scale_factor = max(1.0, np.sqrt(chi2_red))  # Ensure can never shrink credible band, i.e. conservative approach
        # Symmetrically scale the credible interval about the mean
        y_lower = y_pred_mean - (y_pred_mean - y_lower) * scale_factor
        y_upper = y_pred_mean + (y_upper - y_pred_mean) * scale_factor

    return y_pred_mean, y_lower, y_upper


def propagate_forward(x0, sigma_x0, model_function, param_samples, n_samples=2000):
    """
    Propagate uncertainty from x0 through the calibration curve
    using MCMC posterior parameter samples.

    Parameters:
        x0: Central value of x
        sigma_x0: Uncertainty in x
        model_function: Calibration model function f(x, *params)
        param_samples: MCMC posterior samples (n_samples x n_params)
        n_samples: Number of Monte Carlo samples to draw

    Returns:
        y0_samples: Array of y values propagated from x0
    """
    y0_samples = []

    for _ in range(n_samples):
        # Randomly sample parameters from posterior
        params = param_samples[np.random.randint(len(param_samples))]

        # Perturb x0 with Gaussian uncertainty
        x_sample = np.random.normal(x0, sigma_x0)

        # Evaluate model
        y_sample = model_function(x_sample, *params)
        y0_samples.append(y_sample)

    return np.array(y0_samples)

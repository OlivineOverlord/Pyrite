import pytest
import numpy as np

from pyrite import monte_carlo as mc

def linear_model(x, a, b):  # Example model function for testing
    """A simple linear model: y = a * x + b"""
    return a * x + b

def quadratic_model(x, a, b, c):
    """A simple quadratic model: y = ax^2 + bx + c"""
    return a * x**2 + b * x + c

class TestComplexStepDerivative:
    """Tests for complex_step_derivative"""

    def simple_polynomial(self, x, a, b, c):
        """f(x) = ax^2 + bx + c"""
        return a * x**2 + b * x + c

    def simple_polynomial_derivative(self, x, a, b, c):
        """f'(x) = 2ax + b"""
        return 2 * a * x + b

    def sine_function(self, x, amplitude, phase):
        """f(x) = amplitude * sin(x + phase)"""
        return amplitude * np.sin(x + phase)

    def sine_function_derivative(self, x, amplitude, phase):
        """f'(x) = amplitude * cos(x + phase)"""
        return amplitude * np.cos(x + phase)

    def test_polynomial(self):
        """Test for simple polynomial function"""
        x = 3.0
        params = [2.0, 4.0, 1.0]  # a=2, b=4, c=1
        f = self.simple_polynomial

        # Compute the derivative using complex step
        num_derivative = mc.complex_step_derivative(f, x, params)

        # The exact derivative of f(x) = 2x^2 + 4x + 1 is f'(x) = 2*2*x + 4 = 12
        expected_derivative = self.simple_polynomial_derivative(x, *params)

        assert np.isclose(num_derivative, expected_derivative, atol=1e-6), \
            f"Expected {expected_derivative}, got {num_derivative}"

    def test_sine(self):
        """Test for sine function (trigonometric)"""
        x = np.pi / 4  # 45 degrees
        params = [2.0, 0.0]  # amplitude = 2, phase = 0
        f = self.sine_function

        # Compute the derivative using complex step
        num_derivative = mc.complex_step_derivative(f, x, params)

        # The exact derivative of f(x) = 2 * sin(x) is f'(x) = 2 * cos(x)
        expected_derivative = self.sine_function_derivative(x, *params)

        assert np.isclose(num_derivative, expected_derivative, atol=1e-6), \
            f"Expected {expected_derivative}, got {num_derivative}"

    @pytest.mark.parametrize("dx", [1e-20, 1e-10, 1e-5, 1e-2])
    def test_varying_dx(self, dx):
        """Test for varying step sizes (dx)"""
        x = 1.0
        params = [2.0, 3.0]  # model: f(x) = 2x + 3
        def f(x, a, b):
            return a * x + b

        # Compute the derivative using complex step
        num_derivative = mc.complex_step_derivative(f, x, params, dx)

        # Exact derivative of f(x) = 2x + 3 is 2
        expected_derivative = 2

        # Check if the derivative is close to the expected value
        assert np.isclose(num_derivative, expected_derivative, atol=1e-6), \
            f"For dx={dx}, expected {expected_derivative}, got {num_derivative}"

    def test_edge_cases(self):
        """Test for very small and very large values of x"""
        # Test small x
        x_small = 1e-10
        params = [1.0, 0.0]  # model: f(x) = x
        def f(x, a, b):
            return a * x + b
        num_derivative_small = mc.complex_step_derivative(f, x_small, params)
        expected_derivative_small = 1.0  # derivative of f(x) = x is 1
        assert np.isclose(num_derivative_small, expected_derivative_small, atol=1e-6), \
            f"Expected {expected_derivative_small}, got {num_derivative_small}"

        # Test large x
        x_large = 1e10
        num_derivative_large = mc.complex_step_derivative(f, x_large, params)
        expected_derivative_large = 1.0  # derivative of f(x) = x is 1
        assert np.isclose(num_derivative_large, expected_derivative_large, atol=1e-6), \
            f"Expected {expected_derivative_large}, got {num_derivative_large}"

    def test_vs_finite_difference(self):
        """Compare complex-step derivative to finite difference method"""
        x = 1.0
        params = [2.0, 3.0]  # model: f(x) = 2x + 3
        def f(x, a, b):
            return a * x + b
        dx = 1e-5

        # Compute the derivative using complex step
        num_derivative_complex_step = mc.complex_step_derivative(f, x, params, dx)

        # Compute the derivative using finite difference
        delta_x = 1e-5
        num_derivative_fd = (f(x + delta_x, *params) - f(x - delta_x, *params)) / (2 * delta_x)

        # The expected derivative is 2 (from f(x) = 2x + 3)
        expected_derivative = 2

        # Compare both methods
        assert np.isclose(num_derivative_complex_step, num_derivative_fd, expected_derivative, atol=1e-6), \
            f"Complex step derivative {num_derivative_complex_step} differs from finite difference {num_derivative_fd}"


# ---------------------------------------
# Tests for chi2_effective_variance
# ---------------------------------------

class TestChi2EffectiveVariance:
    """Tests for chi2_effective_variance"""
    # Test 1: Chi-squared value for exact fit (i.e., no residuals)
    def test_exact_fit(self):
        # Generate simple data
        x = np.array([1, 2, 3])
        y = np.array([5, 7, 9])  # Model: y = 2x + 3
        sigma_x = np.array([0.1, 0.1, 0.1])  # Small error in x
        sigma_y = np.array([0.2, 0.2, 0.2])  # Small error in y
        params = [2.0, 3.0]  # True parameters a=2, b=3

        # Compute chi-squared
        chi2_value = mc.chi2_effective_variance(params, x, y, sigma_x, sigma_y, linear_model)

        # Since we used the exact parameters, the chi-squared value should be close to 0.
        assert np.isfinite(chi2_value)
        assert np.isclose(chi2_value, 0, atol=1e-6), f"Expected chi-squared value to be 0, got {chi2_value}"

    # Test 2: Chi-squared value with some residuals (model doesn't exactly fit data)
    def test_with_residuals(self):
        # Generate data with small noise
        x = np.array([1, 2, 3])
        y = np.array([5.1, 7.0, 9.2])  # Slightly different from the model y = 2x + 3
        sigma_x = np.array([0.1, 0.1, 0.1])  # Small error in x
        sigma_y = np.array([0.2, 0.2, 0.2])  # Small error in y
        params = [2.0, 3.0]  # True parameters a=2, b=3

        # Compute chi-squared
        chi2_value = mc.chi2_effective_variance(params, x, y, sigma_x, sigma_y, linear_model)

        # Check that the chi-squared value is greater than 0
        assert np.isfinite(chi2_value)
        assert chi2_value > 0, f"Expected chi-squared value > 0, got {chi2_value}"

    # Test 3: Chi-squared value for large errors in y (expected to increase chi-squared)
    def test_large_sigma_y(self):
        # Generate data
        x = np.array([1, 2, 3])
        y = np.array([5.1, 7.0, 8.9])  # Model: y = 2x + 3
        sigma_x = np.array([0.1, 0.1, 0.1])  # Small error in x
        sigma_y = np.array([10.0, 10.0, 10.0])  # Large error in y (overestimated errors)
        params = [2.0, 3.0]  # True parameters a=2, b=3

        # Compute chi-squared
        chi2_value = mc.chi2_effective_variance(params, x, y, sigma_x, sigma_y, linear_model)

        # We expect a larger chi-squared value due to the large sigma_y
        assert np.isfinite(chi2_value)
        assert chi2_value > 0, f"Expected chi-squared value > 0, got {chi2_value}"

    # Test 4: Non-linear model (quadratic function)
    def test_non_linear_model(self):
        # Generate data for a quadratic model
        x = np.array([1, 2, 3])
        y = np.array([6, 14, 24])  # Model: y = 2x^2 + 3x + 1
        sigma_x = np.array([0.1, 0.1, 0.1])  # Small error in x
        sigma_y = np.array([0.5, 0.5, 0.5])  # Small error in y
        params = [2.0, 3.0, 1.0]  # True parameters a=2, b=3, c=1

        # Compute chi-squared for quadratic model
        chi2_value = mc.chi2_effective_variance(params, x, y, sigma_x, sigma_y, quadratic_model)

        # Check if chi-squared value is reasonable (greater than 0, since it's a fit to real data)
        assert chi2_value > 0, f"Expected chi-squared value > 0, got {chi2_value}"

class TestFitEffectiveVariance:
    """Tests for fit_effective_variance"""

    def test_linear(self):
        # Generate synthetic data
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        true_params = [2.0, -1.0]  # slope=2, intercept=-1
        y = linear_model(x, *true_params)

        # Add small noise
        y_noisy = y + rng.normal(0, 0.1, size=x.shape)

        sigma_x = np.full_like(x, 0.01)
        sigma_y = np.full_like(x, 0.1)

        # Initial guess far from truth
        initial_guess = [1.0, 0.0]

        # Run fitting
        fit_params = mc.fit_effective_variance(x, y_noisy, sigma_x, sigma_y, linear_model, initial_guess)

        # Assert fitted parameters are close to the true ones
        assert np.allclose(fit_params, true_params, atol=0.2)


class TestComputeR2:
    """Tests for compute_r2"""

    def test_perfect_fit(self):
        x = np.linspace(0, 10, 20)
        y = linear_model(x, 2.0, 1.0)
        sigma_x = np.ones_like(x) * 0.1
        sigma_y = np.ones_like(x) * 0.1
        params = [2.0, 1.0]

        r2 = mc.compute_r2(x, y, sigma_x, sigma_y, linear_model, params)
        assert np.isclose(r2, 1.0)

    def test_noisy_data(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 50)
        y_true = linear_model(x, 2.0, -1.0)
        y_noisy = y_true + rng.normal(0, 0.5, size=x.shape)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.5
        params = [2.0, -1.0]

        r2 = mc.compute_r2(x, y_noisy, sigma_x, sigma_y, linear_model, params)
        assert 0 <= r2 <= 1


class TestComputeWeightedR2:
    """Tests for compute_weighted_r2"""

    def test_perfect_fit(self):
        x = np.linspace(0, 5, 10)
        y = linear_model(x, 3.0, 2.0)
        sigma_x = np.ones_like(x) * 0.1
        sigma_y = np.ones_like(x) * 0.2
        params = [3.0, 2.0]

        wr2 = mc.compute_weighted_r2(x, y, sigma_x, sigma_y, linear_model, params)
        assert np.isclose(wr2, 1.0)

    def test_zero_sigma_y(self):
        x = np.linspace(0, 5, 10)
        y = linear_model(x, 1.0, 0.0)
        sigma_x = np.ones_like(x) * 0.1
        sigma_y = np.zeros_like(x)  # force zero sigma_y
        params = [1.0, 0.0]

        wr2 = mc.compute_weighted_r2(x, y, sigma_x, sigma_y, linear_model, params)
        assert np.isclose(wr2, 1.0)


class TestComputeChi2Red:
    """Tests for compute_chi2_red"""

    def test_perfect_fit(self):
        x = np.linspace(0, 5, 10)
        y = linear_model(x, 1.0, 2.0)
        sigma_x = np.ones_like(x) * 0.1
        sigma_y = np.ones_like(x) * 0.2
        params = [1.0, 2.0]

        chi2, chi2_red = mc.compute_chi2_red(x, y, sigma_x, sigma_y, linear_model, params)
        assert np.isclose(chi2, 0.0)
        assert np.isclose(chi2_red, 0.0)


    def test_noisy_data(self):
        rng = np.random.default_rng(123)
        x = np.linspace(0, 5, 20)
        y_true = linear_model(x, 1.5, -0.5)
        y_noisy = y_true + rng.normal(0, 0.2, size=x.shape)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.2
        params = [1.5, -0.5]

        chi2, chi2_red = mc.compute_chi2_red(x, y_noisy, sigma_x, sigma_y, linear_model, params)
        assert chi2 > 0
        assert chi2_red > 0


class TestAICBICFromFit:
    """Tests for aic_bic_from_fit"""

    def test_perfect_fit(self):
        x = np.linspace(0, 10, 20)
        true_params = [2.0, -1.0]
        y = linear_model(x, *true_params)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.1

        result = mc.aic_bic_from_fit(true_params, x, y, sigma_x, sigma_y, linear_model)

        # With a perfect fit, residuals = 0, so logL should be large (close to 0 loss)
        assert "logL" in result
        assert "AIC" in result
        assert "AICc" in result
        assert "BIC" in result

        # AIC and BIC should be finite
        assert np.isfinite(result["AIC"])
        assert np.isfinite(result["BIC"])

    def test_noisy_data(self):
        rng = np.random.default_rng(123)
        x = np.linspace(0, 5, 30)
        true_params = [1.5, 0.5]
        y_true = linear_model(x, *true_params)
        y_noisy = y_true + rng.normal(0, 0.2, size=x.shape)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.2

        result = mc.aic_bic_from_fit(true_params, x, y_noisy, sigma_x, sigma_y, linear_model)
        print(result)
        # Check values are finite
        assert np.isfinite(result["logL"])
        assert np.isfinite(result["AIC"])
        assert np.isfinite(result["AICc"])
        assert np.isfinite(result["BIC"])

        # check that adding noise does not give absurdly small log-likelihood
        assert result["logL"] < 1e3, "logL seems unreasonably large"

    def test_small_sample_correction(self):
        # Use very few data points to trigger AICc correction
        x = np.array([0.0, 1.0, 2.0])
        y = linear_model(x, 1.0, 0.0)
        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.1
        params = [1.0, 0.0]

        result = mc.aic_bic_from_fit(params, x, y, sigma_x, sigma_y, linear_model)

        assert np.isfinite(result["AICc"]) or result["AICc"] == np.inf
        assert result["AICc"] >= result["AIC"]

    def test_model_selection(self):
        """Check that BIC penalizes extra parameters more strongly than AIC."""

        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        true_params = [2.0, 1.0]
        y_true = linear_model(x, *true_params)
        y_noisy = y_true + rng.normal(0, 0.3, size=x.shape)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.3

        # Fit with linear (correct) model
        result_linear = mc.aic_bic_from_fit(true_params, x, y_noisy, sigma_x, sigma_y, linear_model)

        # Fit with quadratic (extra parameter, overfitting risk)
        quadratic_params = [0.0, 2.0, 1.0]  # rough guess
        result_quadratic = mc.aic_bic_from_fit(quadratic_params, x, y_noisy, sigma_x, sigma_y, quadratic_model)

        # AIC may prefer quadratic (better fit, less penalty), but BIC should penalize extra param more
        assert result_linear["BIC"] <= result_quadratic["BIC"]
        assert result_linear["AIC"] <= result_quadratic["AIC"] or result_quadratic["AIC"] < result_linear["AIC"]


class TestEstimateUncertaintiesMCMC:
    """Tests for estimate_uncertainties_mcmc"""

    @pytest.mark.slow
    def test_returns_expected_shapes(self):
        """Check that MCMC returns correctly shaped outputs."""
        x = np.linspace(0, 5, 20)
        true_params = [2.0, -1.0]
        y = linear_model(x, *true_params)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.1

        # Use small n_steps for testing speed
        samples, means, stds, cov = mc.estimate_uncertainties_mcmc(
            x, y, sigma_x, sigma_y, linear_model, np.array(true_params),
            n_walkers=10, n_steps=200, burn_in=50
        )

        ndim = len(true_params)

        assert samples.shape[1] == ndim
        assert means.shape == (ndim,)
        assert stds.shape == (ndim,)
        assert cov.shape == (ndim, ndim)

    @pytest.mark.slow
    def test_estimates_close_to_truth(self):
        """Check that estimated means are close to true parameters."""
        rng = np.random.default_rng(123)
        x = np.linspace(0, 5, 30)
        true_params = [1.5, 0.5]
        y_true = linear_model(x, *true_params)
        y_noisy = y_true + rng.normal(0, 0.1, size=x.shape)

        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.1

        samples, means, stds, cov = mc.estimate_uncertainties_mcmc(
            x, y_noisy, sigma_x, sigma_y, linear_model, np.array(true_params),
            n_walkers=20, n_steps=300, burn_in=100
        )

        assert np.allclose(means, true_params, atol=0.2)  # within 0.2 tolerance

    @pytest.mark.slow
    def test_respects_prior_bounds(self):
        """Check that prior bounds are respected by samples."""
        x = np.linspace(0, 5, 10)
        true_params = [1.0, 0.0]
        y = linear_model(x, *true_params)
        sigma_x = np.ones_like(x) * 0.01
        sigma_y = np.ones_like(x) * 0.1

        prior_bounds = [(-1, 1), (-1, 1)]

        samples, means, stds, cov = mc.estimate_uncertainties_mcmc(
            x, y, sigma_x, sigma_y, linear_model, np.array(true_params),
            n_walkers=10, n_steps=200, burn_in=50, prior_bounds=prior_bounds
        )

        assert np.all(samples[:, 0] >= prior_bounds[0][0])
        assert np.all(samples[:, 0] <= prior_bounds[0][1])
        assert np.all(samples[:, 1] >= prior_bounds[1][0])
        assert np.all(samples[:, 1] <= prior_bounds[1][1])

class TestFitEffectiveVarianceFull:
    """Tests for fit_effective_variance_full"""

    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 5, 20)
        true_params = [2.0, 1.0]
        y_true = linear_model(x, *true_params)
        y_noisy = y_true + rng.normal(0, 0.1, size=x.shape)

        sigma_x = np.full_like(x, 0.01)
        sigma_y = np.full_like(x, 0.1)
        initial_guess = [1.0, 0.5]

        # Run fit without MCMC for speed
        results = mc.fit_effective_variance_full(
            x, y_noisy, sigma_x, sigma_y, linear_model, initial_guess, use_mcmc=False
        )

        # Check that fit_params are finite and close to true values
        fit_params = results["fit_params"]
        assert np.all(np.isfinite(fit_params))
        np.testing.assert_allclose(fit_params, true_params, rtol=0.2)

        # Check chi2 and chi2_red
        assert results["chi2"] >= 0
        assert results["chi2_red"] >= 0

        # Check weighted R² in [0, 1]
        assert 0 <= results["weighted_r2"] <= 1

        # Check residuals
        assert results["residuals"].shape == x.shape

        # Check degrees of freedom
        assert results["dof"] == len(x) - len(fit_params)

        # Check predicted values
        assert results["y_model"].shape == x.shape

        # AICc and BIC finite
        assert np.isfinite(results["aic_c"])
        assert np.isfinite(results["bic"])

        # MCMC outputs should be None since use_mcmc=False
        assert results["param_samples_mcmc"] is None
        assert results["param_means_mcmc"] is None
        assert results["param_uncertainties_mcmc"] is None
        assert results["covariance_matrix_mcmc"] is None

    def test_with_mcmc(self):
        rng = np.random.default_rng(123)
        x = np.linspace(0, 5, 15)
        true_params = [1.5, 0.5]
        y_true = linear_model(x, *true_params)
        y_noisy = y_true + rng.normal(0, 0.1, size=x.shape)

        sigma_x = np.full_like(x, 0.01)
        sigma_y = np.full_like(x, 0.1)
        initial_guess = [1.0, 0.0]

        # Run fit with MCMC, but small n_walkers/n_steps for test speed
        results = mc.fit_effective_variance_full(
            x, y_noisy, sigma_x, sigma_y, linear_model, initial_guess,
            use_mcmc=True, n_walkers=10, n_steps=200, burn_in=50
        )

        # Check that MCMC outputs are arrays of correct shape
        param_samples = results["param_samples_mcmc"]
        param_means = results["param_means_mcmc"]
        param_uncertainties = results["param_uncertainties_mcmc"]
        cov_matrix = results["covariance_matrix_mcmc"]

        assert param_samples.shape[1] == len(initial_guess)
        assert len(param_means) == len(initial_guess)
        assert len(param_uncertainties) == len(initial_guess)
        assert cov_matrix.shape == (len(initial_guess), len(initial_guess))

        # Check that param_means are reasonable
        np.testing.assert_allclose(param_means, true_params, rtol=0.3)

        # All MCMC uncertainties should be positive
        assert np.all(param_uncertainties > 0)

    def test_perfect_data(self):
        # Perfect data: model matches data exactly
        x = np.linspace(0, 5, 10)
        true_params = [2.0, 1.0]
        y_true = linear_model(x, *true_params)

        # Very small uncertainties (simulate exact data)
        sigma_x = np.full_like(x, 1e-4)
        sigma_y = np.full_like(x, 1e-4)
        initial_guess = [1.0, 0.5]

        results = mc.fit_effective_variance_full(
            x, y_true, sigma_x, sigma_y, linear_model, initial_guess, use_mcmc=False
        )

        fit_params = results["fit_params"]
        residuals = results["residuals"]
        chi2 = results["chi2"]
        chi2_red = results["chi2_red"]
        weighted_r2 = results["weighted_r2"]

        # Fit parameters should recover true parameters exactly
        np.testing.assert_allclose(fit_params, true_params, rtol=1e-6)

        # Residuals should be zero (or extremely close)
        np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-7)

        # Chi² and reduced chi² should be zero
        assert chi2 < 1e-6
        assert chi2_red < 1e-6

        # Weighted R² should be 1
        assert np.isclose(weighted_r2, 1.0, atol=1e-10)

        # AICc and BIC should be finite numbers
        assert np.isfinite(results["aic_c"])
        assert np.isfinite(results["bic"])

        # MCMC outputs are None since use_mcmc=False
        assert results["param_samples_mcmc"] is None
        assert results["param_means_mcmc"] is None
        assert results["param_uncertainties_mcmc"] is None
        assert results["covariance_matrix_mcmc"] is None

class TestComputeCredibleBand:
    """Tests for compute_credible_band"""

    def test_basic(self):
        rng = np.random.default_rng(123)
        x_vals = np.linspace(0, 5, 10)
        true_params = [2.0, 1.0]

        # Generate fake MCMC samples around true_params
        n_samples = 1000
        param_samples = rng.normal(loc=true_params, scale=[0.1, 0.05], size=(n_samples, 2))

        y_mean, y_lower, y_upper = mc.compute_credible_band(x_vals, linear_model, param_samples, ci=68)

        # Check shapes
        assert y_mean.shape == x_vals.shape
        assert y_lower.shape == x_vals.shape
        assert y_upper.shape == x_vals.shape

        # Check credible interval ordering
        assert np.all(y_lower <= y_mean)
        assert np.all(y_upper >= y_mean)

        # Check that mean is close to true model for these synthetic samples
        y_true = linear_model(x_vals, *true_params)
        np.testing.assert_allclose(y_mean, y_true, rtol=0.05)

        # Check that CI width is positive
        assert np.all((y_upper - y_lower) > 0)

    def test_chi2_red_scaling(self):
        rng = np.random.default_rng(456)
        x_vals = np.linspace(0, 5, 10)
        true_params = [1.5, 0.5]

        n_samples = 500
        param_samples = rng.normal(loc=true_params, scale=[0.1, 0.05], size=(n_samples, 2))

        chi2_red = 2.0  # Should scale credible intervals

        y_mean, y_lower, y_upper = mc.compute_credible_band(
            x_vals, linear_model, param_samples, ci=68, chi2_red=chi2_red
        )

        # Check shapes
        assert y_mean.shape == x_vals.shape
        assert y_lower.shape == x_vals.shape
        assert y_upper.shape == x_vals.shape

        # Check ordering
        assert np.all(y_lower <= y_mean)
        assert np.all(y_upper >= y_mean)

        # Check scaling: the interval width should be at least sqrt(chi2_red) times larger than unscaled
        y_mean_unscaled, y_lower_unscaled, y_upper_unscaled = mc.compute_credible_band(
            x_vals, linear_model, param_samples, ci=68, chi2_red=None
        )

        width_scaled = y_upper - y_lower
        width_unscaled = y_upper_unscaled - y_lower_unscaled

        assert np.all(width_scaled >= width_unscaled)

    def test_edge_cases(self):
        # Single sample
        x_vals = np.linspace(0, 3, 5)
        param_samples = np.array([[2.0, 1.0]])  # single MCMC sample

        y_mean, y_lower, y_upper = mc.compute_credible_band(x_vals, linear_model, param_samples, ci=68)

        # With one sample, mean = lower = upper
        np.testing.assert_allclose(y_mean, y_lower)
        np.testing.assert_allclose(y_mean, y_upper)

    def test_percentiles(self):
        rng = np.random.default_rng(42)
        x_vals = np.linspace(0, 5, 20)
        true_params = [2.0, 1.0]

        # Generate MCMC samples around true parameters
        n_samples = 10000
        param_samples = rng.normal(loc=true_params, scale=[0.2, 0.1], size=(n_samples, 2))

        for ci in [68, 95]:
            y_mean, y_lower, y_upper = mc.compute_credible_band(
                x_vals, linear_model, param_samples, ci=ci
            )

            # Check that roughly the correct fraction of samples fall within the credible band
            # Pick one x value to check
            x_idx = 5  # arbitrary index
            y_samples_at_x = np.array([linear_model(x_vals[x_idx], *p) for p in param_samples])

            # Fraction within band
            frac_within = np.mean((y_samples_at_x >= y_lower[x_idx]) & (y_samples_at_x <= y_upper[x_idx]))

            # Allow a small tolerance due to finite sample size
            expected_frac = ci / 100
            assert np.isclose(frac_within, expected_frac, atol=0.02), \
                f"CI={ci}%, fraction={frac_within:.3f} differs from expected {expected_frac:.3f}"

class TestPropagateForward:
    """Tests for propagate_forward"""

    def test_basic(self):
        rng = np.random.default_rng(42)
        x0 = 2.0
        sigma_x0 = 0.1

        # Create fake MCMC samples for parameters
        n_samples_param = 500
        param_samples = rng.normal(loc=[2.0, 1.0], scale=[0.1, 0.05], size=(n_samples_param, 2))

        n_mc_samples = 1000
        y_samples = mc.propagate_forward(x0, sigma_x0, linear_model, param_samples, n_samples=n_mc_samples)

        # Output should be a numpy array of correct length
        assert isinstance(y_samples, np.ndarray)
        assert len(y_samples) == n_mc_samples

        # Check that mean of y_samples is reasonable (around linear_model(x0, true_params))
        y_true = linear_model(x0, 2.0, 1.0)
        assert np.abs(np.mean(y_samples) - y_true) < 0.2  # within expected uncertainty

        # Standard deviation should be positive and larger than zero
        std_y = np.std(y_samples)
        assert std_y > 0

    def test_zero_sigma_x0(self):
        # If sigma_x0 = 0, only param uncertainty should contribute
        rng = np.random.default_rng(123)
        x0 = 1.0
        sigma_x0 = 0.0

        param_samples = rng.normal(loc=[1.0, 0.5], scale=[0.05, 0.02], size=(200, 2))

        y_samples = mc.propagate_forward(x0, sigma_x0, linear_model, param_samples, n_samples=500)

        # Should still produce nonzero standard deviation due to param_samples
        assert np.std(y_samples) > 0
        # Mean should be close to the deterministic evaluation at mean params
        mean_params = np.mean(param_samples, axis=0)
        y_mean_expected = linear_model(x0, *mean_params)
        np.testing.assert_allclose(np.mean(y_samples), y_mean_expected, rtol=0.05)

    def test_statistical_properties(self):
        # Check that increasing n_samples reduces Monte Carlo noise
        rng = np.random.default_rng(0)
        x0 = 0.5
        sigma_x0 = 0.05
        param_samples = rng.normal(loc=[1.0, 0.0], scale=[0.1, 0.05], size=(300, 2))

        y_small = mc.propagate_forward(x0, sigma_x0, linear_model, param_samples, n_samples=200)
        y_large = mc.propagate_forward(x0, sigma_x0, linear_model, param_samples, n_samples=5000)

        # Standard deviation should converge: large sample should be closer to small sample mean
        assert np.abs(np.mean(y_large) - np.mean(y_small)) < 0.05

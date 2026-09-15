//! Coulomb wave functions and the auxiliary API used by their representations.

use crate::numerics::{
    digamma_complex, gamma_complex, kummer_m, kummer_m_regularized, log_gamma_complex,
    reciprocal_gamma, tricomi_u,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Coulomb parameter `1 / (a k)`.
pub fn eta(a: f64, k: f64) -> f64 {
    assert!(a != 0.0 && k != 0.0, "a and k must be nonzero");
    1.0 / (a * k)
}

/// Coulomb parameter `1 / (a k)` for complex inputs.
pub fn eta_complex(a: Complex64, k: Complex64) -> Complex64 {
    assert!(a != Complex64::new(0.0, 0.0), "a must be nonzero");
    assert!(k != Complex64::new(0.0, 0.0), "k must be nonzero");
    1.0 / (a * k)
}

/// Coulomb parameter `1 / sqrt(epsilon)`.
pub fn eta_from_energy(epsilon: f64) -> f64 {
    assert!(epsilon != 0.0, "epsilon must be nonzero");
    1.0 / epsilon.sqrt()
}

/// Coulomb parameter `1 / sqrt(epsilon)` for a complex energy.
pub fn eta_from_energy_complex(epsilon: Complex64) -> Complex64 {
    assert!(
        epsilon != Complex64::new(0.0, 0.0),
        "epsilon must be nonzero"
    );
    1.0 / epsilon.sqrt()
}

/// Analytic Coulomb normalization for complex parameters.
pub fn normalization_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let imaginary = Complex64::i();
    let log_gamma_product = 0.5
        * (log_gamma_complex(ell + 1.0 + imaginary * eta)
            + log_gamma_complex(ell + 1.0 - imaginary * eta));
    (ell * (2.0_f64).ln() - PI * eta / 2.0 + log_gamma_product - log_gamma_complex(2.0 * ell + 2.0))
        .exp()
}

/// Positive Coulomb normalization for real parameters.
pub fn normalization(ell: f64, eta: f64) -> f64 {
    normalization_complex(ell.into(), eta.into()).re
}

/// Coulomb phase for real parameters and positive `rho`.
pub fn phase(ell: f64, eta: f64, rho: f64) -> f64 {
    assert!(rho > 0.0, "rho must be positive");
    rho - ell * PI / 2.0 - eta * (2.0 * rho).ln()
        + log_gamma_complex(Complex64::new(ell + 1.0, eta)).im
}

/// Coulomb phase generalized to complex parameters.
pub fn phase_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    let log_gamma_imaginary = log_gamma_complex(ell + 1.0 + Complex64::i() * eta).im;
    rho - ell * PI / 2.0 - eta * (2.0 * rho).ln() + log_gamma_imaginary
}

/// Regular Coulomb wave function for complex inputs.
pub fn f_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    let imaginary = Complex64::i();
    normalization_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (-imaginary * rho).exp()
        * kummer_m(
            ell + 1.0 - imaginary * eta,
            2.0 * ell + 2.0,
            2.0 * imaginary * rho,
        )
}

/// Regular Coulomb wave function for real inputs.
pub fn f(ell: f64, eta: f64, rho: f64) -> f64 {
    if ell == 0.0 && eta == 0.0 {
        return rho.sin();
    }
    f_complex(ell.into(), eta.into(), rho.into()).re
}

/// Outgoing Coulomb normalization for complex parameters.
pub fn d_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    Complex64::new(0.0, -2.0).powc(2.0 * ell + 1.0)
        * gamma_complex(ell + 1.0 + Complex64::i() * eta)
        * reciprocal_gamma(2.0 * ell + 2.0)
        / normalization_complex(ell, eta)
}

/// Outgoing Coulomb normalization for real parameters.
pub fn d_plus(ell: f64, eta: f64) -> Complex64 {
    d_plus_complex(ell.into(), eta.into())
}

/// Incoming Coulomb normalization for complex parameters.
pub fn d_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    Complex64::new(0.0, 2.0).powc(2.0 * ell + 1.0)
        * gamma_complex(ell + 1.0 - Complex64::i() * eta)
        * reciprocal_gamma(2.0 * ell + 2.0)
        / normalization_complex(ell, eta)
}

/// Incoming Coulomb normalization for real parameters.
pub fn d_minus(ell: f64, eta: f64) -> Complex64 {
    d_minus_complex(ell.into(), eta.into())
}

/// Outgoing Coulomb wave for complex inputs.
pub fn h_plus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    let imaginary = Complex64::i();
    d_plus_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (imaginary * rho).exp()
        * tricomi_u(
            ell + 1.0 + imaginary * eta,
            2.0 * ell + 2.0,
            -2.0 * imaginary * rho,
        )
}

fn h_plus_asymptotic_real(ell: f64, eta: f64, rho: f64) -> (Complex64, Complex64) {
    let a = Complex64::new(ell + 1.0, eta);
    let b = 2.0 * ell + 2.0;
    let z = Complex64::new(0.0, -2.0 * rho);
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    let mut smallest_term = term.norm();
    let mut best = (sum, derivative_sum);

    for n in 1..=1000 {
        let previous = (n - 1) as f64;
        term *= -(a + previous) * (a - b + n as f64) / (n as f64 * z);
        sum += term;
        derivative_sum -= n as f64 * term / rho;
        if term.norm() < smallest_term {
            smallest_term = term.norm();
            best = (sum, derivative_sum);
        } else {
            (sum, derivative_sum) = best;
            break;
        }
        if term.norm() <= f64::EPSILON * sum.norm().max(1.0) / 16.0 {
            break;
        }
    }

    let prefactor =
        d_plus(ell, eta) * rho.powf(ell + 1.0) * Complex64::from_polar(1.0, rho) * z.powc(-a);
    (
        prefactor * sum,
        prefactor * ((Complex64::i() - Complex64::i() * eta / rho) * sum + derivative_sum),
    )
}

fn advance_coulomb(
    ell: f64,
    eta: f64,
    rho: f64,
    value: Complex64,
    derivative: Complex64,
    step: f64,
) -> (Complex64, Complex64) {
    let acceleration = |position: f64, amplitude: Complex64| {
        (ell * (ell + 1.0) / position.powi(2) + 2.0 * eta / position - 1.0) * amplitude
    };
    let k1_value = derivative;
    let k1_derivative = acceleration(rho, value);
    let k2_value = derivative + step * k1_derivative / 2.0;
    let k2_derivative = acceleration(rho + step / 2.0, value + step * k1_value / 2.0);
    let k3_value = derivative + step * k2_derivative / 2.0;
    let k3_derivative = acceleration(rho + step / 2.0, value + step * k2_value / 2.0);
    let k4_value = derivative + step * k3_derivative;
    let k4_derivative = acceleration(rho + step, value + step * k3_value);
    (
        value + step * (k1_value + 2.0 * k2_value + 2.0 * k3_value + k4_value) / 6.0,
        derivative
            + step * (k1_derivative + 2.0 * k2_derivative + 2.0 * k3_derivative + k4_derivative)
                / 6.0,
    )
}

fn irregular_real(ell: f64, eta: f64, rho: f64) -> f64 {
    assert!(
        ell.is_finite() && eta.is_finite() && rho.is_finite(),
        "Coulomb parameters must be finite"
    );
    if rho <= 0.0 {
        return (0.5
            * (h_plus_complex(ell.into(), eta.into(), rho.into())
                + h_minus_complex(ell.into(), eta.into(), rho.into())))
        .re;
    }

    // At the starting radius the Poincaré series is far into its stable
    // regime for ordinary f64 parameters. Integrating the real ODE backward
    // avoids the removable singularity of the Tricomi connection formula at
    // every integer b = 2ℓ+2.
    let mut position = rho.max(20.0 + 3.0 * (ell.abs() + eta.abs()));
    let (mut value, mut derivative) = h_plus_asymptotic_real(ell, eta, position);
    let mut steps = 0_usize;
    while position > rho {
        let step_size = 0.003_f64.min(position / 300.0).min(position - rho);
        let step = -step_size;
        (value, derivative) = advance_coulomb(ell, eta, position, value, derivative, step);
        position += step;
        steps += 1;
        assert!(steps <= 2_000_000, "Coulomb ODE work limit exceeded");
    }
    value.re
}

/// Outgoing Coulomb wave for real inputs.
pub fn h_plus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    Complex64::new(irregular_real(ell, eta, rho), f(ell, eta, rho))
}

/// Incoming Coulomb wave for complex inputs.
pub fn h_minus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    let imaginary = Complex64::i();
    d_minus_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (-imaginary * rho).exp()
        * tricomi_u(
            ell + 1.0 - imaginary * eta,
            2.0 * ell + 2.0,
            2.0 * imaginary * rho,
        )
}

/// Incoming Coulomb wave for real inputs.
pub fn h_minus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    h_plus(ell, eta, rho).conj()
}

/// Regular Coulomb wave reconstructed from outgoing and incoming waves.
pub fn f_imag(ell: f64, eta: f64, rho: f64) -> f64 {
    f(ell, eta, rho)
}

/// Regular Coulomb wave reconstructed from complex outgoing and incoming waves.
pub fn f_imag_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    (h_plus_complex(ell, eta, rho) - h_minus_complex(ell, eta, rho)) / (2.0 * Complex64::i())
}

/// Irregular Coulomb wave for real inputs.
pub fn g(ell: f64, eta: f64, rho: f64) -> f64 {
    irregular_real(ell, eta, rho)
}

/// Irregular Coulomb wave for complex inputs.
pub fn g_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    0.5 * (h_plus_complex(ell, eta, rho) + h_minus_complex(ell, eta, rho))
}

/// Regularized confluent hypergeometric function `M(a,b,z) / Gamma(b)`.
pub fn m_regularized(alpha: Complex64, beta: Complex64, argument: Complex64) -> Complex64 {
    kummer_m_regularized(alpha, beta, argument)
}

/// Modified Coulomb function for complex inputs.
pub fn phi_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    (2.0 * eta * rho).powc(ell + 1.0)
        * (Complex64::i() * rho).exp()
        * m_regularized(
            ell + 1.0 + Complex64::i() * eta,
            2.0 * ell + 2.0,
            -2.0 * Complex64::i() * rho,
        )
}

/// Modified Coulomb function for real inputs.
pub fn phi(ell: f64, eta: f64, rho: f64) -> Complex64 {
    phi_complex(ell.into(), eta.into(), rho.into())
}

/// Gamow auxiliary product for nonnegative integer or half-integer `ell`.
pub fn gamow_w(ell: f64, eta: f64) -> f64 {
    gamow_w_complex(ell, eta.into()).re
}

/// Gamow auxiliary product with a complex Coulomb parameter.
pub fn gamow_w_complex(ell: f64, eta: Complex64) -> Complex64 {
    assert!(eta != Complex64::new(0.0, 0.0), "eta must be nonzero");
    assert!(
        ell >= 0.0 && 2.0 * ell == (2.0 * ell).round(),
        "ell must be a nonnegative integer or half-integer"
    );
    let start = if ell == ell.round() { 0.0 } else { 0.5 };
    let mut result = Complex64::new(1.0, 0.0);
    let mut j = start;
    while j <= ell {
        result *= 1.0 + j * j / (eta * eta);
        j += 1.0;
    }
    result
}

/// Complex `w+` Coulomb auxiliary function.
pub fn w_plus(ell: f64, eta: f64) -> Complex64 {
    w_plus_complex(ell.into(), eta.into())
}

/// Complex `w+` Coulomb auxiliary function for complex parameters.
pub fn w_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let i_eta = Complex64::i() * eta;
    gamma_complex(ell + 1.0 + Complex64::i() * eta)
        / (i_eta.powc(2.0 * ell + 1.0) * gamma_complex(-ell + Complex64::i() * eta))
}

/// Complex `w-` Coulomb auxiliary function.
pub fn w_minus(ell: f64, eta: f64) -> Complex64 {
    w_minus_complex(ell.into(), eta.into())
}

/// Complex `w-` Coulomb auxiliary function for complex parameters.
pub fn w_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let minus_i_eta = -Complex64::i() * eta;
    gamma_complex(ell + 1.0 - Complex64::i() * eta)
        / (minus_i_eta.powc(2.0 * ell + 1.0) * gamma_complex(-ell - Complex64::i() * eta))
}

/// Complex `h+` auxiliary (distinct from the outgoing wave [`h_plus`]).
pub fn aux_h_plus(ell: f64, eta: f64) -> Complex64 {
    aux_h_plus_complex(ell.into(), eta.into())
}

/// Complex `h+` auxiliary for complex parameters.
pub fn aux_h_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let i_eta = Complex64::i() * eta;
    0.5 * (digamma_complex(ell + 1.0 + i_eta) + digamma_complex(-ell + i_eta)) - i_eta.ln()
}

/// Complex `h-` auxiliary (distinct from the incoming wave [`h_minus`]).
pub fn aux_h_minus(ell: f64, eta: f64) -> Complex64 {
    aux_h_minus_complex(ell.into(), eta.into())
}

/// Complex `h-` auxiliary for complex parameters.
pub fn aux_h_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let minus_i_eta = -Complex64::i() * eta;
    0.5 * (digamma_complex(ell + 1.0 + minus_i_eta) + digamma_complex(-ell + minus_i_eta))
        - minus_i_eta.ln()
}

/// Real digamma-based Coulomb auxiliary function.
pub fn aux_g(ell: f64, eta: f64) -> f64 {
    aux_g_complex(ell.into(), eta.into())
}

/// Real-valued `g` auxiliary evaluated with complex parameters.
pub fn aux_g_complex(ell: Complex64, eta: Complex64) -> f64 {
    assert!(eta != Complex64::new(0.0, 0.0), "eta must be nonzero");
    (0.5 * (digamma_complex(ell + 1.0 + Complex64::i() * eta)
        + digamma_complex(ell + 1.0 - Complex64::i() * eta)))
    .re - eta.norm().ln()
}

fn finite_difference_step(step: Option<f64>) -> f64 {
    let step = step.unwrap_or(f64::EPSILON.cbrt());
    assert!(
        step.is_finite() && step > 0.0,
        "finite-difference step must be positive"
    );
    step
}

/// Central order derivative of [`phi`].
pub fn phi_dot(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> Complex64 {
    let step = finite_difference_step(step);
    (phi(ell + step, eta, rho) - phi(ell - step, eta, rho)) / (2.0 * step)
}

/// Central order derivative of [`phi_complex`].
pub fn phi_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> Complex64 {
    let step = finite_difference_step(step);
    (phi_complex(ell + step, eta, rho) - phi_complex(ell - step, eta, rho)) / (2.0 * step)
}

/// Central order derivative of [`f`].
pub fn f_dot(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> f64 {
    let step = finite_difference_step(step);
    (f(ell + step, eta, rho) - f(ell - step, eta, rho)) / (2.0 * step)
}

/// Central order derivative of [`f_complex`].
pub fn f_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> Complex64 {
    let step = finite_difference_step(step);
    (f_complex(ell + step, eta, rho) - f_complex(ell - step, eta, rho)) / (2.0 * step)
}

/// Coulomb auxiliary `Psi`.
pub fn psi(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> Complex64 {
    let step = Some(finite_difference_step(step));
    gamow_w(ell, eta) * phi_dot(ell, eta, rho, step) / 2.0
        + phi_dot(-ell - 1.0, eta, rho, step) / 2.0
}

/// Coulomb auxiliary `Psi` with complex `eta` and `rho`.
pub fn psi_complex(ell: f64, eta: Complex64, rho: Complex64, step: Option<f64>) -> Complex64 {
    let step = Some(finite_difference_step(step));
    gamow_w_complex(ell, eta) * phi_dot_complex(ell.into(), eta, rho, step) / 2.0
        + phi_dot_complex((-ell - 1.0).into(), eta, rho, step) / 2.0
}

/// Coulomb auxiliary `I`.
pub fn i(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> Complex64 {
    normalization(ell, eta) * libm::tgamma(2.0 * ell + 2.0) / (2.0 * eta).powf(ell + 1.0)
        * psi(ell, eta, rho, step)
}

/// Coulomb auxiliary `I` with complex `eta` and `rho`.
pub fn i_complex(ell: f64, eta: Complex64, rho: Complex64, step: Option<f64>) -> Complex64 {
    normalization_complex(ell.into(), eta) * gamma_complex((2.0 * ell + 2.0).into())
        / (2.0 * eta).powf(ell + 1.0)
        * psi_complex(ell, eta, rho, step)
}

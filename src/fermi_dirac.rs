use std::f64::consts::FRAC_PI_2;

/// F_j(x) for any order `j > -1`.
///
/// For `x <= -1` the series `F_j(x) = Γ(j+1) Σ_{k≥1} (−1)^{k+1} e^{kx} / k^{j+1}`
/// converges geometrically. Above that the defining integral is evaluated by
/// double-exponential quadrature: tanh-sinh on `[0, x]` and exp-sinh on
/// `[x, ∞)` for `x > 0`, exp-sinh on `[0, ∞)` otherwise. The integrands are
/// written in logarithmic form, so the `t^j` endpoint singularity and the
/// exponential tail neither overflow nor underflow before they are weighted.
fn general_order(j: f64, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= -1.0 {
        return libm::tgamma(j + 1.0) * alternating_series(j, x);
    }
    let order = j + 1.0;
    if x > 0.0 {
        // [0, x] with t = x σ(y), x − t = x σ(−y), y = π sinh s and σ the
        // logistic function; t^j dt = x^{j+1} σ(y)^{j+1} σ(−y) π cosh(s) ds.
        let inner = double_exponential(|s| {
            let y = std::f64::consts::PI * s.sinh();
            let log_sigma = -softplus(-y);
            let log_complement = -softplus(y);
            let distance = x * log_complement.exp();
            std::f64::consts::PI * s.cosh() * (order * log_sigma + log_complement).exp()
                / ((-distance).exp() + 1.0)
        });
        // [x, ∞) with t = x + w, w = exp(y), y = (π/2) sinh s.
        let outer = double_exponential(|s| {
            let y = FRAC_PI_2 * s.sinh();
            let w = y.exp();
            if !w.is_finite() {
                return 0.0;
            }
            FRAC_PI_2 * s.cosh() * (y + j * (x + w).ln() - w - softplus(-w)).exp()
        });
        x.powf(order) * inner + outer
    } else {
        // [0, ∞) with t = exp(y), y = (π/2) sinh s; t − x ≥ 0.
        double_exponential(|s| {
            let y = FRAC_PI_2 * s.sinh();
            let excess = y.exp() - x;
            if !excess.is_finite() {
                return 0.0;
            }
            FRAC_PI_2 * s.cosh() * (order * y - excess - softplus(-excess)).exp()
        })
    }
}

/// `Σ_{k≥1} (−1)^{k+1} e^{kx} / k^{j+1}`, for `x <= -1`.
fn alternating_series(j: f64, x: f64) -> f64 {
    let ratio = x.exp();
    let mut power = ratio;
    let mut sum = 0.0;
    let mut k = 1.0_f64;
    loop {
        let term = power / k.powf(j + 1.0);
        sum += if k % 2.0 == 1.0 { term } else { -term };
        if term <= 0.25 * f64::EPSILON * sum.abs() {
            return sum;
        }
        k += 1.0;
        power *= ratio;
    }
}

/// `ln(1 + e^z)` without overflow or loss of accuracy.
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// `∫ f(s) ds` over the real line for an integrand that decays double
/// exponentially, by the trapezoidal rule with step halving.
///
/// The error of the rule roughly squares with each halving, so once two levels
/// agree to 1e-10 the finer one is accurate to rounding.
fn double_exponential(f: impl Fn(f64) -> f64) -> f64 {
    // Beyond |s| = 8 every integrand here is below the f64 range.
    const S_MAX: f64 = 8.0;
    const LEVELS: usize = 10;
    // Adds f(offset + k·spacing) for k = 0, 1, 2, … in one direction until
    // the terms no longer change `reference`.
    let ray = |offset: f64, spacing: f64, reference: f64| {
        let mut sum = 0.0;
        let mut s = offset;
        while s.abs() <= S_MAX {
            let value = f(s);
            sum += value;
            if value.abs() <= 1e-2 * f64::EPSILON * (reference + sum).abs() && s != offset {
                break;
            }
            s += spacing;
        }
        sum
    };

    let mut step = 0.5;
    let centre = f(0.0);
    let mut nodes = centre + ray(step, step, centre) + ray(-step, -step, centre);
    let mut estimate = step * nodes;
    for _ in 0..LEVELS {
        // The new nodes are the odd multiples of step/2.
        let half = 0.5 * step;
        nodes += ray(half, step, nodes) + ray(-half, -step, nodes);
        step = half;
        let refined = step * nodes;
        if (refined - estimate).abs() <= 1e-10 * refined.abs() {
            return refined;
        }
        estimate = refined;
    }
    panic!("Fermi-Dirac quadrature did not converge")
}

/// Complete Fermi-Dirac integral F_j(x) = ∫₀^∞ tʲ / (exp(t-x) + 1) dt.
///
/// Arises in solid-state physics as a measure of carrier statistics in
/// semiconductors; the index `j` is typically a half-integer related to the
/// dimensionality of the band structure.
///
/// Uses the exact formula ln(1 + eˣ) for j = 0. Every other j uses the
/// alternating series in eˣ for x ≤ −1 and double-exponential quadrature of
/// the defining integral otherwise.
///
/// # Accuracy
///
/// Measured against mpmath for j ∈ [−0.5, 10] and x ∈ [−50, 10⁴] (see the
/// crate README): relative error below 10⁻¹⁴.
///
/// # Panics
/// Panics if `j < -0.5`.
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::fermi_dirac_integral;
///
/// // F_0(0) = ln(2) exactly
/// let v = fermi_dirac_integral(0.0, 0.0);
/// assert!((v - std::f64::consts::LN_2).abs() < 1e-14);
///
/// // F_{1/2}(0) ≈ 0.6781 (tabulated value)
/// let v = fermi_dirac_integral(0.5, 0.0);
/// assert!((v - 0.6781).abs() < 1e-4);
/// ```
pub fn fermi_dirac_integral(j: f64, x: f64) -> f64 {
    assert!(j >= -0.5, "j must be >= -0.5, got {j}");

    if j == 0.0 {
        softplus(x)
    } else {
        general_order(j, x)
    }
}

/// Normalized Fermi-Dirac integral F̃_j(x) = F_j(x) / Γ(j+1).
///
/// Dividing by Γ(j+1) gives a function that approaches eˣ as x → −∞ for
/// every order, and x^{j+1}/Γ(j+2) in the degenerate limit x → +∞. Useful when
/// comparing Fermi-Dirac statistics across different orders.
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::fermi_dirac_integral_norm;
///
/// // F̃_{1/2}(0) ≈ 0.7652
/// let v = fermi_dirac_integral_norm(0.5, 0.0);
/// assert!((v - 0.7652).abs() < 1e-4);
/// ```
pub fn fermi_dirac_integral_norm(j: f64, x: f64) -> f64 {
    fermi_dirac_integral(j, x) / libm::tgamma(j + 1.0)
}

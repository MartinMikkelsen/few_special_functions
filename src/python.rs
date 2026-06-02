use pyo3::prelude::*;

// ── Clausen ──────────────────────────────────────────────────────────────────

/// Clausen function Cl_n(θ) for n = 1..6 using N = 10 quadrature nodes.
#[pyfunction]
fn clausen(n: usize, theta: f64) -> f64 {
    crate::clausen::clausen(n, theta)
}

/// Clausen function Cl_n(θ) using N = 20 nodes (extended precision).
#[pyfunction]
fn clausen_n20(n: usize, theta: f64) -> f64 {
    crate::clausen::clausen_n20(n, theta)
}

// ── Fermi-Dirac ───────────────────────────────────────────────────────────────

/// Complete Fermi-Dirac integral F_j(x) = ∫₀^∞ tʲ / (exp(t-x) + 1) dt.
#[pyfunction]
fn fermi_dirac_integral(j: f64, x: f64) -> f64 {
    crate::fermi_dirac::fermi_dirac_integral(j, x)
}

/// Normalised Fermi-Dirac integral F_j(x) / Γ(j+1).
#[pyfunction]
fn fermi_dirac_integral_norm(j: f64, x: f64) -> f64 {
    crate::fermi_dirac::fermi_dirac_integral_norm(j, x)
}

// ── Debye ─────────────────────────────────────────────────────────────────────

/// Generalized Debye function D_n(β, x).
#[pyfunction]
fn debye_function(n: f64, beta: f64, x: f64) -> f64 {
    crate::debye::debye_function(n, beta, x)
}

// ── Fresnel ───────────────────────────────────────────────────────────────────

/// Fresnel cosine integral C(x) = ∫₀ˣ cos(π/2 · t²) dt.
#[pyfunction]
fn fresnel_c(x: f64) -> f64 {
    crate::fresnel::fresnel_c(x)
}

/// Fresnel sine integral S(x) = ∫₀ˣ sin(π/2 · t²) dt.
#[pyfunction]
fn fresnel_s(x: f64) -> f64 {
    crate::fresnel::fresnel_s(x)
}

/// Fresnel integrals — returns (C(x), S(x)) as a Python tuple.
#[pyfunction]
fn fresnel(x: f64) -> (f64, f64) {
    let (c, s, _) = crate::fresnel::fresnel(x);
    (c, s)
}

// ── Marcum Q ──────────────────────────────────────────────────────────────────

/// Generalized Marcum Q-function Q_μ(a, b).
#[pyfunction]
fn marcum_q(mu: f64, a: f64, b: f64) -> f64 {
    crate::marcum_q::marcum_q(mu, a, b)
}

/// Derivative ∂Q_M(a,b)/∂b of the Marcum Q-function.
#[pyfunction]
fn dq_db(m: u32, a: f64, b: f64) -> f64 {
    crate::marcum_q::dq_db(m, a, b)
}

// ── Module registration ───────────────────────────────────────────────────────

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clausen, m)?)?;
    m.add_function(wrap_pyfunction!(clausen_n20, m)?)?;
    m.add_function(wrap_pyfunction!(fermi_dirac_integral, m)?)?;
    m.add_function(wrap_pyfunction!(fermi_dirac_integral_norm, m)?)?;
    m.add_function(wrap_pyfunction!(debye_function, m)?)?;
    m.add_function(wrap_pyfunction!(fresnel_c, m)?)?;
    m.add_function(wrap_pyfunction!(fresnel_s, m)?)?;
    m.add_function(wrap_pyfunction!(fresnel, m)?)?;
    m.add_function(wrap_pyfunction!(marcum_q, m)?)?;
    m.add_function(wrap_pyfunction!(dq_db, m)?)?;
    Ok(())
}

use num_complex::Complex64;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn checked<T>(operation: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("invalid arguments");
        PyValueError::new_err(message.to_owned())
    })
}

#[pyfunction]
fn clausen(n: usize, theta: f64) -> PyResult<f64> {
    checked(|| crate::clausen::clausen(n, theta))
}

#[pyfunction]
fn clausen_n20(n: usize, theta: f64) -> PyResult<f64> {
    checked(|| crate::clausen::clausen_n20(n, theta))
}

#[pyfunction(signature = (n, theta, nodes=10, m=20))]
fn clausen_with_options(n: usize, theta: f64, nodes: usize, m: usize) -> PyResult<f64> {
    checked(|| crate::clausen::clausen_with_options(n, theta, nodes, m))
}

#[pyfunction]
fn ci_complex(z: Complex64) -> Complex64 {
    crate::clausen::ci_complex(z)
}

#[pyfunction]
fn f_clausen(n: usize, z: Complex64, theta: f64) -> Complex64 {
    crate::clausen::f_clausen(n, z, theta)
}

#[pyfunction]
fn f_n(n: usize, k: usize, theta: f64) -> f64 {
    crate::clausen::f_n(n, k, theta)
}

#[pyfunction]
fn bose_einstein_integral(k: f64, eta: f64) -> PyResult<f64> {
    checked(|| crate::bose_einstein::bose_einstein_integral(k, eta))
}

#[pyfunction]
fn bose_einstein_integral_norm(k: f64, eta: f64) -> PyResult<f64> {
    checked(|| crate::bose_einstein::bose_einstein_integral_norm(k, eta))
}

#[pyfunction]
fn fermi_dirac_integral(j: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::fermi_dirac::fermi_dirac_integral(j, x))
}

#[pyfunction]
fn fermi_dirac_integral_norm(j: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::fermi_dirac::fermi_dirac_integral_norm(j, x))
}

#[pyfunction]
fn debye_function(n: f64, beta: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::debye::debye_function(n, beta, x))
}

#[pyfunction]
fn debye_function_order_one(beta: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::debye::debye_function_order_one(beta, x))
}

#[pyfunction(signature = (n, beta, x, tol=1e-35, max_terms=2000))]
fn debye_function_tol(n: f64, beta: f64, x: f64, tol: f64, max_terms: usize) -> PyResult<f64> {
    checked(|| crate::debye::debye_function_tol(n, beta, x, tol, max_terms))
}

#[pyfunction]
fn fresnel(x: f64) -> (f64, f64, Complex64) {
    crate::fresnel::fresnel(x)
}

#[pyfunction]
fn fresnel_complex(z: Complex64) -> (Complex64, Complex64, Complex64) {
    crate::fresnel::fresnel_complex(z)
}

#[pyfunction]
fn fresnel_c(x: f64) -> f64 {
    crate::fresnel::fresnel_c(x)
}

#[pyfunction]
fn fresnel_s(x: f64) -> f64 {
    crate::fresnel::fresnel_s(x)
}

#[pyfunction]
fn fresnel_e(x: f64) -> Complex64 {
    crate::fresnel::fresnel_e(x)
}

#[pyfunction]
fn fresnel_c_complex(z: Complex64) -> Complex64 {
    crate::fresnel::fresnel_c_complex(z)
}

#[pyfunction]
fn fresnel_s_complex(z: Complex64) -> Complex64 {
    crate::fresnel::fresnel_s_complex(z)
}

#[pyfunction]
fn fresnel_e_complex(z: Complex64) -> Complex64 {
    crate::fresnel::fresnel_e_complex(z)
}

#[pyfunction]
fn dawson(x: f64) -> f64 {
    crate::dawson::dawson(x)
}

#[pyfunction]
fn voigt(x: f64, y: f64) -> PyResult<f64> {
    checked(|| crate::voigt::voigt(x, y))
}

#[pyfunction]
fn marcum_q(mu: f64, a: f64, b: f64) -> PyResult<f64> {
    checked(|| crate::marcum_q::marcum_q(mu, a, b))
}

#[pyfunction]
fn marcum_q_order_one(a: f64, b: f64) -> PyResult<f64> {
    checked(|| crate::marcum_q::marcum_q_order_one(a, b))
}

#[pyfunction]
fn dq_db(m: u32, a: f64, b: f64) -> PyResult<f64> {
    checked(|| crate::marcum_q::dq_db(m, a, b))
}

#[pyfunction]
fn dq_db_order_one(a: f64, b: f64) -> PyResult<f64> {
    checked(|| crate::marcum_q::dq_db_order_one(a, b))
}

macro_rules! cylinder_binding {
    ($name:ident) => {
        #[pyfunction]
        fn $name(a: f64, x: f64) -> PyResult<f64> {
            checked(|| crate::parabolic_cylinder::$name(a, x))
        }
    };
}
cylinder_binding!(u);
cylinder_binding!(v);
cylinder_binding!(w);
cylinder_binding!(du);
cylinder_binding!(dv);
cylinder_binding!(dw);
cylinder_binding!(u_scaled);
cylinder_binding!(v_scaled);

#[pyfunction]
fn parabolic_cylinder_d(nu: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::parabolic_cylinder::parabolic_cylinder_d(nu, x))
}

#[pyfunction]
fn d_parabolic_cylinder_d(nu: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::parabolic_cylinder::d_parabolic_cylinder_d(nu, x))
}

#[pyfunction]
fn parabolic_cylinder_d_scaled(nu: f64, x: f64) -> PyResult<f64> {
    checked(|| crate::parabolic_cylinder::parabolic_cylinder_d_scaled(nu, x))
}

macro_rules! whittaker_real_binding {
    ($name:ident) => {
        #[pyfunction]
        fn $name(kappa: f64, mu: f64, z: f64) -> PyResult<f64> {
            checked(|| crate::whittaker::$name(kappa, mu, z))
        }
    };
}
whittaker_real_binding!(whittaker_m);
whittaker_real_binding!(whittaker_w);
whittaker_real_binding!(d_whittaker_m);
whittaker_real_binding!(d_whittaker_w);

macro_rules! whittaker_complex_binding {
    ($name:ident) => {
        #[pyfunction]
        fn $name(kappa: Complex64, mu: Complex64, z: Complex64) -> PyResult<Complex64> {
            checked(|| crate::whittaker::$name(kappa, mu, z))
        }
    };
}
whittaker_complex_binding!(whittaker_m_complex);
whittaker_complex_binding!(whittaker_w_complex);
whittaker_complex_binding!(d_whittaker_m_complex);
whittaker_complex_binding!(d_whittaker_w_complex);

#[pyfunction]
fn coulomb_eta(a: f64, k: f64) -> PyResult<f64> {
    checked(|| crate::coulomb::eta(a, k))
}

#[pyfunction]
fn coulomb_eta_complex(a: Complex64, k: Complex64) -> PyResult<Complex64> {
    checked(|| crate::coulomb::eta_complex(a, k))
}

#[pyfunction]
fn coulomb_eta_from_energy(epsilon: f64) -> PyResult<f64> {
    checked(|| crate::coulomb::eta_from_energy(epsilon))
}

#[pyfunction]
fn coulomb_eta_from_energy_complex(epsilon: Complex64) -> PyResult<Complex64> {
    checked(|| crate::coulomb::eta_from_energy_complex(epsilon))
}

#[pyfunction]
fn coulomb_normalization(ell: f64, eta: f64) -> f64 {
    crate::coulomb::normalization(ell, eta)
}

#[pyfunction]
fn coulomb_normalization_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::normalization_complex(ell, eta)
}

#[pyfunction]
fn coulomb_phase(ell: f64, eta: f64, rho: f64) -> PyResult<f64> {
    checked(|| crate::coulomb::phase(ell, eta, rho))
}

#[pyfunction]
fn coulomb_phase_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::phase_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_f(ell: f64, eta: f64, rho: f64) -> f64 {
    crate::coulomb::f(ell, eta, rho)
}

#[pyfunction]
fn coulomb_f_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::f_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_d_plus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::d_plus(ell, eta)
}

#[pyfunction]
fn coulomb_d_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::d_plus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_d_minus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::d_minus(ell, eta)
}

#[pyfunction]
fn coulomb_d_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::d_minus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_h_plus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    crate::coulomb::h_plus(ell, eta, rho)
}

#[pyfunction]
fn coulomb_h_plus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::h_plus_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_h_minus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    crate::coulomb::h_minus(ell, eta, rho)
}

#[pyfunction]
fn coulomb_h_minus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::h_minus_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_f_imag(ell: f64, eta: f64, rho: f64) -> f64 {
    crate::coulomb::f_imag(ell, eta, rho)
}

#[pyfunction]
fn coulomb_f_imag_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::f_imag_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_g(ell: f64, eta: f64, rho: f64) -> f64 {
    crate::coulomb::g(ell, eta, rho)
}

#[pyfunction]
fn coulomb_g_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::g_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_m_regularized(alpha: Complex64, beta: Complex64, argument: Complex64) -> Complex64 {
    crate::coulomb::m_regularized(alpha, beta, argument)
}

#[pyfunction]
fn coulomb_phi(ell: f64, eta: f64, rho: f64) -> Complex64 {
    crate::coulomb::phi(ell, eta, rho)
}

#[pyfunction]
fn coulomb_phi_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    crate::coulomb::phi_complex(ell, eta, rho)
}

#[pyfunction]
fn coulomb_w(ell: f64, eta: f64) -> PyResult<f64> {
    checked(|| crate::coulomb::gamow_w(ell, eta))
}

#[pyfunction]
fn coulomb_w_complex(ell: f64, eta: Complex64) -> PyResult<Complex64> {
    checked(|| crate::coulomb::gamow_w_complex(ell, eta))
}

#[pyfunction]
fn coulomb_w_plus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::w_plus(ell, eta)
}

#[pyfunction]
fn coulomb_w_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::w_plus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_w_minus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::w_minus(ell, eta)
}

#[pyfunction]
fn coulomb_w_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::w_minus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_aux_h_plus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::aux_h_plus(ell, eta)
}

#[pyfunction]
fn coulomb_aux_h_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::aux_h_plus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_aux_h_minus(ell: f64, eta: f64) -> Complex64 {
    crate::coulomb::aux_h_minus(ell, eta)
}

#[pyfunction]
fn coulomb_aux_h_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    crate::coulomb::aux_h_minus_complex(ell, eta)
}

#[pyfunction]
fn coulomb_aux_g(ell: f64, eta: f64) -> PyResult<f64> {
    checked(|| crate::coulomb::aux_g(ell, eta))
}

#[pyfunction]
fn coulomb_aux_g_complex(ell: Complex64, eta: Complex64) -> PyResult<f64> {
    checked(|| crate::coulomb::aux_g_complex(ell, eta))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_phi_dot(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> PyResult<Complex64> {
    checked(|| crate::coulomb::phi_dot(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_phi_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> PyResult<Complex64> {
    checked(|| crate::coulomb::phi_dot_complex(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_f_dot(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> PyResult<f64> {
    checked(|| crate::coulomb::f_dot(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_f_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> PyResult<Complex64> {
    checked(|| crate::coulomb::f_dot_complex(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_psi(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> PyResult<Complex64> {
    checked(|| crate::coulomb::psi(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_psi_complex(
    ell: f64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> PyResult<Complex64> {
    checked(|| crate::coulomb::psi_complex(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_i(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> PyResult<Complex64> {
    checked(|| crate::coulomb::i(ell, eta, rho, step))
}

#[pyfunction(signature = (ell, eta, rho, step=None))]
fn coulomb_i_complex(
    ell: f64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> PyResult<Complex64> {
    checked(|| crate::coulomb::i_complex(ell, eta, rho, step))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add_functions {
        ($($function:ident),+ $(,)?) => {$({
            m.add_function(wrap_pyfunction!($function, m)?)?;
        })+};
    }
    add_functions!(
        clausen,
        clausen_n20,
        clausen_with_options,
        ci_complex,
        f_clausen,
        f_n,
        bose_einstein_integral,
        bose_einstein_integral_norm,
        fermi_dirac_integral,
        fermi_dirac_integral_norm,
        debye_function,
        debye_function_order_one,
        debye_function_tol,
        fresnel,
        fresnel_complex,
        fresnel_c,
        fresnel_s,
        fresnel_e,
        fresnel_c_complex,
        fresnel_s_complex,
        fresnel_e_complex,
        dawson,
        voigt,
        marcum_q,
        marcum_q_order_one,
        dq_db,
        dq_db_order_one,
        u,
        v,
        w,
        du,
        dv,
        dw,
        u_scaled,
        v_scaled,
        parabolic_cylinder_d,
        d_parabolic_cylinder_d,
        parabolic_cylinder_d_scaled,
        whittaker_m,
        whittaker_m_complex,
        whittaker_w,
        whittaker_w_complex,
        d_whittaker_m,
        d_whittaker_m_complex,
        d_whittaker_w,
        d_whittaker_w_complex,
        coulomb_eta,
        coulomb_eta_complex,
        coulomb_eta_from_energy,
        coulomb_eta_from_energy_complex,
        coulomb_normalization,
        coulomb_normalization_complex,
        coulomb_phase,
        coulomb_phase_complex,
        coulomb_f,
        coulomb_f_complex,
        coulomb_d_plus,
        coulomb_d_plus_complex,
        coulomb_d_minus,
        coulomb_d_minus_complex,
        coulomb_h_plus,
        coulomb_h_plus_complex,
        coulomb_h_minus,
        coulomb_h_minus_complex,
        coulomb_f_imag,
        coulomb_f_imag_complex,
        coulomb_g,
        coulomb_g_complex,
        coulomb_m_regularized,
        coulomb_phi,
        coulomb_phi_complex,
        coulomb_w,
        coulomb_w_complex,
        coulomb_w_plus,
        coulomb_w_plus_complex,
        coulomb_w_minus,
        coulomb_w_minus_complex,
        coulomb_aux_h_plus,
        coulomb_aux_h_plus_complex,
        coulomb_aux_h_minus,
        coulomb_aux_h_minus_complex,
        coulomb_aux_g,
        coulomb_aux_g_complex,
        coulomb_phi_dot,
        coulomb_phi_dot_complex,
        coulomb_f_dot,
        coulomb_f_dot_complex,
        coulomb_psi,
        coulomb_psi_complex,
        coulomb_i,
        coulomb_i_complex,
    );
    Ok(())
}

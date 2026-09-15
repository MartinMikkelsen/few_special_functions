use few_special_functions::whittaker::{
    d_whittaker_m, d_whittaker_m_complex, d_whittaker_w, d_whittaker_w_complex, whittaker_m,
    whittaker_m_complex, whittaker_w, whittaker_w_complex,
};
use num_complex::Complex64;

fn relative(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert!(
        (actual - expected).norm() <= tolerance * expected.norm().max(1e-300),
        "expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn real_values_match_julia() {
    relative(
        whittaker_m(0.7, 0.4, 1.3).into(),
        Complex64::new(0.7916208751360724, 0.0),
        3e-12,
    );
    relative(
        whittaker_w(0.7, 0.4, 1.3).into(),
        Complex64::new(0.6784443734133974, 0.0),
        3e-12,
    );
    relative(
        d_whittaker_w(0.7, 0.4, 1.3).into(),
        Complex64::new(-0.009663021577526138, 0.0),
        8e-11,
    );

    relative(
        whittaker_m(-10.0, 0.0, 0.001).into(),
        Complex64::new(0.031939797803498295, 0.0),
        3e-12,
    );
    relative(
        whittaker_w(-10.0, 0.0, 0.001).into(),
        Complex64::new(9.780249037297617e-8, 0.0),
        2e-10,
    );
    relative(
        whittaker_w(10.0, 0.0, 0.001).into(),
        Complex64::new(38739.319626828445, 0.0),
        2e-12,
    );
    relative(
        d_whittaker_w(10.0, 0.0, 0.001).into(),
        Complex64::new(7457984.93024646, 0.0),
        2e-12,
    );
}

#[test]
fn complex_values_match_julia() {
    let kappa = Complex64::new(0.2, 0.0);
    let mu = Complex64::new(0.3, 0.0);
    let z = Complex64::new(1.0, 2.0);
    relative(
        whittaker_m_complex(kappa, mu, z),
        Complex64::new(0.9761524750899391, 1.0524124716674081),
        3e-12,
    );
    relative(
        whittaker_w_complex(kappa, mu, z),
        Complex64::new(0.507200798454674, -0.5003214235598513),
        3e-11,
    );
    relative(
        d_whittaker_m_complex(kappa, mu, z),
        Complex64::new(0.2347214911456684, 0.02922463446455343),
        3e-12,
    );
    relative(
        d_whittaker_w_complex(kappa, mu, z),
        Complex64::new(-0.27333808117393815, 0.18957179096115767),
        8e-11,
    );

    let kappa = Complex64::new(0.2, 0.1);
    let mu = Complex64::new(0.3, -0.2);
    let z = Complex64::new(0.0, 20.0);
    relative(
        whittaker_m_complex(kappa, mu, z),
        Complex64::new(0.4304752734280515, -3.0384725169636857),
        2e-12,
    );
    relative(
        whittaker_w_complex(kappa, mu, z),
        Complex64::new(-1.549924294571801, -0.06186269209089626),
        2e-12,
    );
    relative(
        d_whittaker_m_complex(kappa, mu, z),
        Complex64::new(-0.6083783148029401, 1.7086661995413106),
        2e-12,
    );
    relative(
        d_whittaker_w_complex(kappa, mu, z),
        Complex64::new(0.7666818312211366, 0.04636687992030583),
        2e-12,
    );
}

#[test]
fn elementary_and_large_argument_cases() {
    for z in [0.1, 1.0, 100.0] {
        relative(
            whittaker_m(0.0, 0.5, z).into(),
            Complex64::new(2.0 * (z / 2.0).sinh(), 0.0),
            3e-12,
        );
        relative(
            whittaker_w(0.0, 0.5, z).into(),
            Complex64::new((-z / 2.0).exp(), 0.0),
            3e-12,
        );
        relative(
            d_whittaker_w(0.0, 0.5, z).into(),
            Complex64::new(-0.5 * (-z / 2.0).exp(), 0.0),
            3e-12,
        );
    }
    relative(
        whittaker_m(-10.0, 0.0, 1000.0).into(),
        Complex64::new(1.354414987203051e241, 0.0),
        3e-10,
    );
    relative(
        whittaker_w(-10.0, 0.0, 1000.0).into(),
        Complex64::new(6.388445079995855e-248, 0.0),
        3e-10,
    );
}

#[test]
#[should_panic]
fn real_api_rejects_negative_argument() {
    whittaker_w(0.2, 0.3, -1.0);
}

#[test]
#[should_panic]
fn m_rejects_parameter_pole() {
    d_whittaker_m(0.2, -0.5, 1.0);
}

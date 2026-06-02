mod table;

use few_special_functions::{
    clausen::clausen, debye::debye_function, fermi_dirac::fermi_dirac_integral, fresnel::fresnel,
    marcum_q::marcum_q,
};
use std::f64::consts::PI;

fn main() {
    // ── Clausen ──────────────────────────────────────────────────────────────
    table::header("Clausen  Cl_n(θ)");
    table::col_headers(&["n=2", "n=3", "n=4"]);
    for i in 0..=6 {
        let theta = i as f64 * PI / 3.0;
        table::row(
            &format!("θ = {i}π/3"),
            &[clausen(2, theta), clausen(3, theta), clausen(4, theta)],
        );
    }

    // ── Fermi-Dirac ───────────────────────────────────────────────────────────
    table::header("Fermi-Dirac  F_j(x)");
    table::col_headers(&["j=−½", "j=0", "j=½", "j=3/2"]);
    for x in [-2.0_f64, 0.0, 1.0, 3.0, 5.0] {
        table::row(
            &format!("x = {x:>4.1}"),
            &[
                fermi_dirac_integral(-0.5, x),
                fermi_dirac_integral(0.0, x),
                fermi_dirac_integral(0.5, x),
                fermi_dirac_integral(1.5, x),
            ],
        );
    }

    // ── Debye ─────────────────────────────────────────────────────────────────
    table::header("Debye  D_n(β=1, x)");
    table::col_headers(&["n=1", "n=2", "n=3", "n=5"]);
    for x in [0.5_f64, 1.0, 2.0, 4.0, 8.0] {
        table::row(
            &format!("x = {x:>4.1}"),
            &[
                debye_function(1.0, 1.0, x),
                debye_function(2.0, 1.0, x),
                debye_function(3.0, 1.0, x),
                debye_function(5.0, 1.0, x),
            ],
        );
    }

    // ── Fresnel ───────────────────────────────────────────────────────────────
    table::header("Fresnel  (C(x), S(x))");
    table::col_headers(&["C(x)", "S(x)"]);
    for x in [0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let (c, s, _) = fresnel(x);
        table::row(&format!("x = {x:>4.1}"), &[c, s]);
    }

    // ── Marcum Q ──────────────────────────────────────────────────────────────
    table::header("Marcum Q  Q_μ(a=1, b)");
    table::col_headers(&["μ=1", "μ=2", "μ=5"]);
    for b in [0.5_f64, 1.0, 1.5, 2.0, 3.0] {
        table::row(
            &format!("b = {b:>4.1}"),
            &[
                marcum_q(1.0, 1.0, b),
                marcum_q(2.0, 1.0, b),
                marcum_q(5.0, 1.0, b),
            ],
        );
    }
}

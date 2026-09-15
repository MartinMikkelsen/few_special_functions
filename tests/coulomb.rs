use few_special_functions::coulomb::{
    aux_g, aux_g_complex, aux_h_minus, aux_h_minus_complex, aux_h_plus, aux_h_plus_complex,
    d_minus, d_minus_complex, d_plus, d_plus_complex, eta, eta_complex, eta_from_energy,
    eta_from_energy_complex, f, f_complex, f_dot, f_dot_complex, f_imag, f_imag_complex, g,
    g_complex, gamow_w, gamow_w_complex, h_minus, h_minus_complex, h_plus, h_plus_complex, i,
    i_complex, m_regularized, normalization, normalization_complex, phase, phase_complex, phi,
    phi_complex, phi_dot, phi_dot_complex, psi, psi_complex, w_minus, w_minus_complex, w_plus,
    w_plus_complex,
};
use num_complex::Complex64;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.16e}, got {actual:.16e}"
    );
}

#[test]
fn normalization_and_parameter_helpers_match_julia() {
    assert_eq!(eta(2.0, 0.5), 1.0);
    close(eta_from_energy(4.0), 0.5, 1e-15);
    close(normalization(0.0, 0.0), 1.0, 2e-14);
    close(normalization(1.0, 0.0), 1.0 / 3.0, 2e-14);
    close(
        normalization(0.0, -500.0),
        (1000.0 * std::f64::consts::PI).sqrt(),
        2e-11,
    );
    assert!(phase(1.0, 0.2, 1.5).is_finite());
}

#[test]
fn regular_and_irregular_waves_match_julia() {
    let cases = [
        (0.0, 0.5, 1.0, 0.5166015003141811, 1.1974869707984859),
        (1.0, 1.0, 2.0, 0.36141285774505333, 1.7652192403391787),
        (1.5, -0.3, 3.5, 0.9176292951006562, -0.4661805522113568),
        (0.0, -2.0, 0.1, 0.28764960791219424, 0.3179738412003934),
    ];
    for (ell, charge, rho, expected_f, expected_g) in cases {
        close(f(ell, charge, rho), expected_f, 2e-8);
        close(g(ell, charge, rho), expected_g, 2e-6);
        close(f_imag(ell, charge, rho), expected_f, 2e-7);
        let outgoing = h_plus(ell, charge, rho);
        let incoming = h_minus(ell, charge, rho);
        assert!((outgoing.conj() - incoming).norm() < 2e-7);
        assert!((outgoing - Complex64::new(expected_g, expected_f)).norm() < 2e-6);
    }
    for rho in [0.1, 1.0, 2.5] {
        close(f(0.0, 0.0, rho), rho.sin(), 2e-13);
        close(
            f_complex(0.0.into(), 0.0.into(), rho.into()).re,
            rho.sin(),
            2e-13,
        );
    }

    // The most singular point in the Julia CoulombG reference grid. This
    // guards the integer-b Tricomi limit used by real half-integer orders.
    close(g(2.0, 2.0, 0.1), 13007.786583811625, 1e-5);
}

#[test]
fn complex_api_matches_julia() {
    let ell = Complex64::new(0.4, 0.2);
    let charge = Complex64::new(0.2, 0.3);
    let rho = Complex64::new(1.2, 0.1);
    let close_complex = |actual: Complex64, expected: Complex64| {
        assert!(
            (actual - expected).norm() <= 3e-13 * expected.norm().max(1.0),
            "expected {expected:?}, got {actual:?}"
        );
    };
    let close_finite_difference = |actual: Complex64, expected: Complex64| {
        assert!(
            (actual - expected).norm() <= 3e-9 * expected.norm().max(1.0),
            "expected {expected:?}, got {actual:?}"
        );
    };
    close_complex(
        normalization_complex(ell, charge),
        Complex64::new(0.38431518943324905, -0.3552916355806418),
    );
    close_complex(
        phase_complex(ell, charge, rho),
        Complex64::new(0.2699701230845533, -0.4944661993695235),
    );
    close_complex(
        f_complex(ell, charge, rho),
        Complex64::new(0.6259616696629275, -0.23519498804542094),
    );
    close_complex(
        d_plus_complex(ell, charge),
        Complex64::new(-4.335358450301599, -4.921105014284012),
    );
    close_complex(
        d_minus_complex(ell, charge),
        Complex64::new(-1.8549701671917165, -0.7219149363481772),
    );
    close_complex(
        h_plus_complex(ell, charge, rho),
        Complex64::new(1.2421536508766287, 0.998437959758962),
    );
    close_complex(
        h_minus_complex(ell, charge, rho),
        Complex64::new(0.7717636747857851, -0.2534853795668933),
    );
    close_complex(
        g_complex(ell, charge, rho),
        Complex64::new(1.0069586628312068, 0.37247629009603433),
    );
    close_complex(
        phi_complex(ell, charge, rho),
        Complex64::new(0.08610350320025523, 0.4007308748983482),
    );

    let charge = Complex64::new(0.5, 0.2);
    close_complex(
        eta_complex(Complex64::new(2.0, 0.3), Complex64::new(0.5, -0.1)),
        Complex64::new(0.9685913108895993, 0.047018995674252405),
    );
    close_complex(
        eta_from_energy_complex(Complex64::new(4.0, 0.3)),
        Complex64::new(0.4989496183897817, -0.018684372636118645),
    );
    close_complex(
        gamow_w_complex(1.0, charge),
        Complex64::new(3.497027348394768, -2.3781212841854935),
    );
    close_complex(
        w_plus_complex(ell, charge),
        Complex64::new(1.8162750411159503, -0.10683140227867224),
    );
    close_complex(
        w_minus_complex(ell, charge),
        Complex64::new(1.6091863166692328, 0.014199928311128143),
    );
    close_complex(
        aux_h_plus_complex(ell, charge),
        Complex64::new(0.47432227218948075, -0.4955031791991047),
    );
    close_complex(
        aux_h_minus_complex(ell, charge),
        Complex64::new(0.7323362590481134, -0.14580071955516893),
    );
    close(aux_g_complex(ell, charge), 0.6953128074141738, 3e-13);
    close_finite_difference(
        phi_dot_complex(ell, charge, rho, Some(1e-5)),
        Complex64::new(-1.5346124067228126, -0.7324876615533159),
    );
    close_finite_difference(
        f_dot_complex(ell, charge, rho, Some(1e-5)),
        Complex64::new(-0.4304791661469131, 0.1122137106318588),
    );
    close_finite_difference(
        psi_complex(1.0, charge, rho, Some(1e-5)),
        Complex64::new(-0.4077716830099569, -1.083245548519231),
    );
    close_finite_difference(
        i_complex(1.0, charge, rho, Some(1e-5)),
        Complex64::new(-0.8481562669467176, -0.06183677285482975),
    );
    close_complex(
        f_imag_complex(ell, charge, rho),
        Complex64::new(0.4405796364727688, -0.14534347763048183),
    );
}

#[test]
fn auxiliary_api_is_complete_and_consistent() {
    let plus = d_plus(1.0, 1.2);
    let minus = d_minus(1.0, 1.2);
    assert!((plus.conj() - minus).norm() < 2e-12);
    assert!((w_plus(1.5, -1.2).conj() - w_minus(1.5, -1.2)).norm() < 2e-12);
    assert!((aux_h_plus(1.5, 1.2).conj() - aux_h_minus(1.5, 1.2)).norm() < 2e-12);
    assert!(aux_g(1.5, 1.2).is_finite());
    close(gamow_w(2.0, 1.0), 10.0, 1e-14);
    close(gamow_w(0.5, 2.0), 1.0625, 1e-14);

    let regularized = m_regularized(2.0.into(), 3.0.into(), 4.0.into());
    close(regularized.re, 10.299653131214534, 2e-13);
    assert!(regularized.im.abs() < 2e-14);
    assert!(phi(1.0, 0.5, 2.0).im.abs() < 2e-12);
}

#[test]
fn finite_difference_helpers_are_consistent() {
    let (ell, charge, rho, step) = (1.0, 0.5, 2.0, 1e-5);
    let expected_phi = (phi(ell + step, charge, rho) - phi(ell - step, charge, rho)) / (2.0 * step);
    assert!((phi_dot(ell, charge, rho, Some(step)) - expected_phi).norm() < 1e-12);
    let expected_f = (f(ell + step, charge, rho) - f(ell - step, charge, rho)) / (2.0 * step);
    close(f_dot(ell, charge, rho, Some(step)), expected_f, 1e-12);
    assert!(psi(1.0, charge, rho, None).re.is_finite());
    assert!(i(1.0, charge, rho, None).re.is_finite());
}

#[test]
#[should_panic]
fn gamow_rejects_other_orders() {
    gamow_w(0.3, 1.0);
}

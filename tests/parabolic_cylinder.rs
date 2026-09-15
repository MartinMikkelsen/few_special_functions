use few_special_functions::parabolic_cylinder::{
    d_parabolic_cylinder_d, du, dv, dw, parabolic_cylinder_d, parabolic_cylinder_d_scaled, u,
    u_scaled, v, v_scaled, w,
};

fn relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1e-300),
        "expected {expected:.16e}, got {actual:.16e}"
    );
}

#[test]
fn function_and_derivative_values_match_julia() {
    let a = -1.25459881152638;
    let x = 5.70351922786027;
    relative(u(a, x), 0.0010961750819519622, 2e-12);
    relative(v(a, x), 139.1535424172683, 2e-12);
    relative(w(a, x), 0.3139466789175475, 2e-11);
    relative(du(a, x), -0.002982047814472406, 2e-12);
    relative(dv(a, x), 349.32562331145743, 2e-12);
    relative(dw(a, x), 1.4186664284579742, 2e-11);
    relative(w(0.1, 8.00001), -0.205949151506127, 2e-11);
}

#[test]
fn negative_argument_and_gaussian_special_case() {
    relative(
        u(4.50714306409916, -6.00652435683281),
        1316297.50250584,
        2e-12,
    );
    for x in [-20.0, -7.0, 10.0, 20.0] {
        relative(u(-0.5, x), (-x * x / 4.0).exp(), 3e-13);
        relative(du(-0.5, x), -x * (-x * x / 4.0).exp() / 2.0, 3e-13);
    }

    // Values from the Julia package's MATLAB reference grid. These exercise
    // negative arguments where the positive-integrand U representation becomes
    // sharply peaked and the convergent series is the stable f64 path.
    for (a, x, expected) in [
        (-3.017782, -3.065662, -1.1073029265740295),
        (-4.070111, -3.560917, 2.20371105821678),
        (-0.292368, -1.533182, 1.2372602234253007),
    ] {
        assert!((u(a, x) - expected).abs() <= 1e-5);
    }
}

#[test]
fn scaled_values_match_julia() {
    let references = [
        (0.0, 0.0, 1.2162802142575202, 0.6862126275593261),
        (10.0, 5.0, 0.351065573357017, 0.28189727651781193),
        (-10.0, 7.0, 0.5387138787924095, 0.5069454033737099),
        (0.0, 60.0, 0.12908600517683427, 0.10301719023914642),
        (1000.0, 1.0, 0.12573785240006705, 0.10032050746551777),
        (-1000.0, 1.0, 0.13779780667374855, 0.08393566783235636),
        (-100.0, 20.0, 0.6061866107661307, 0.41902126450472393),
    ];
    for (a, x, expected_u, expected_v) in references {
        relative(u_scaled(a, x), expected_u, 3e-11);
        relative(v_scaled(a, x), expected_v, 3e-11);
        assert_eq!(parabolic_cylinder_d_scaled(-a - 0.5, x), u_scaled(a, x));
    }
}

#[test]
fn standard_d_wrappers_match_u() {
    for (nu, x) in [(0.0, 2.0), (1.0, 2.0), (-2.5, -1.0)] {
        assert_eq!(parabolic_cylinder_d(nu, x), u(-nu - 0.5, x));
        assert_eq!(d_parabolic_cylinder_d(nu, x), du(-nu - 0.5, x));
    }
}

#[test]
#[should_panic]
fn scaled_rejects_negative_x() {
    u_scaled(0.0, -1.0);
}

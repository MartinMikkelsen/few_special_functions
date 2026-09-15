import math

import few_special_functions as fsf


def close(actual, expected, tolerance=1e-11):
    assert abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def test_python_api_parity():
    close(fsf.bose_einstein_integral_norm(0.5, -1.0), 0.4284407345998379)
    close(fsf.debye_function_order_one(1.0, 1.0), fsf.debye_function(1.0, 1.0, 1.0))
    close(fsf.fresnel_e_complex(1 + 1j), 0.49390555890759856 * (1 + 1j))
    close(fsf.marcum_q_order_one(1.2, 1.6), fsf.marcum_q(1.0, 1.2, 1.6))
    close(fsf.parabolic_cylinder_d(1.0, 2.0), 2.0 / math.e)
    close(fsf.whittaker_w(0.0, 0.5, 2.0), 1.0 / math.e)
    close(fsf.whittaker_w_complex(0j, 0.5 + 0j, 1 + 2j), cmath_exp(-(1 + 2j) / 2))
    close(fsf.coulomb_f(0.0, 0.0, 1.0), math.sin(1.0))
    close(fsf.coulomb_eta_complex(2 + 0.3j, 0.5 - 0.1j), 0.9685913108895993 + 0.047018995674252405j)


def cmath_exp(value):
    import cmath

    return cmath.exp(value)


def test_all_family_entry_points_exist():
    names = {
        "clausen", "clausen_n20", "clausen_with_options", "ci_complex",
        "f_clausen", "f_n",
        "bose_einstein_integral", "bose_einstein_integral_norm",
        "fermi_dirac_integral", "fermi_dirac_integral_norm",
        "debye_function", "debye_function_order_one", "debye_function_tol",
        "fresnel", "fresnel_complex", "fresnel_c", "fresnel_s", "fresnel_e",
        "fresnel_c_complex", "fresnel_s_complex", "fresnel_e_complex",
        "dawson", "voigt", "marcum_q", "marcum_q_order_one", "dq_db",
        "dq_db_order_one", "u", "v", "w", "du", "dv", "dw", "u_scaled",
        "v_scaled", "parabolic_cylinder_d", "d_parabolic_cylinder_d",
        "parabolic_cylinder_d_scaled",
        "whittaker_m", "whittaker_w", "d_whittaker_m", "d_whittaker_w",
        "whittaker_m_complex", "whittaker_w_complex", "d_whittaker_m_complex",
        "d_whittaker_w_complex", "coulomb_eta", "coulomb_eta_complex",
        "coulomb_eta_from_energy", "coulomb_eta_from_energy_complex",
        "coulomb_normalization", "coulomb_normalization_complex", "coulomb_phase",
        "coulomb_phase_complex", "coulomb_f", "coulomb_f_complex",
        "coulomb_d_plus", "coulomb_d_plus_complex", "coulomb_d_minus",
        "coulomb_d_minus_complex", "coulomb_h_plus", "coulomb_h_plus_complex",
        "coulomb_h_minus", "coulomb_h_minus_complex", "coulomb_f_imag",
        "coulomb_f_imag_complex",
        "coulomb_g", "coulomb_g_complex", "coulomb_m_regularized", "coulomb_phi",
        "coulomb_phi_complex", "coulomb_w", "coulomb_w_complex",
        "coulomb_w_plus", "coulomb_w_plus_complex", "coulomb_w_minus",
        "coulomb_w_minus_complex", "coulomb_aux_h_plus",
        "coulomb_aux_h_plus_complex", "coulomb_aux_h_minus",
        "coulomb_aux_h_minus_complex", "coulomb_aux_g", "coulomb_aux_g_complex",
        "coulomb_phi_dot", "coulomb_phi_dot_complex", "coulomb_f_dot",
        "coulomb_f_dot_complex", "coulomb_psi", "coulomb_psi_complex",
        "coulomb_i", "coulomb_i_complex",
    }
    assert names <= set(dir(fsf))

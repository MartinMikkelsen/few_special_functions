use few_special_functions::clausen::clausen;
use few_special_functions::fermi_dirac::{fermi_dirac_integral, fermi_dirac_integral_norm};

fn main() {
    // F_0(0) = ln(2) ≈ 0.6931
    println!("F_0(0)   = {:.6}", fermi_dirac_integral(0.0, 0.0));
    // F_{1/2}(0) ≈ 0.6781
    println!("F_½(0)   = {:.6}", fermi_dirac_integral(0.5, 0.0));
    // F_{3/2}(1) ≈ 1.8673
    println!("F_3/2(1) = {:.6}", fermi_dirac_integral(1.5, 1.0));
    // Normalised
    println!("F̃_½(0)  = {:.6}", fermi_dirac_integral_norm(0.5, 0.0));

    // Cl_2(π/3) — a well-known value: Cl_2(π/3) = Catalan-related ≈ 1.01494
    println!("Cl_2(π/3) = {:.6}", clausen(2, std::f64::consts::PI / 3.0));
    // Cl_2(π/2) ≈ 0.91597 (Catalan's constant G)
    println!("Cl_2(π/2) = {:.6} (≈ Catalan G ≈ 0.915966)", clausen(2, std::f64::consts::PI / 2.0));
    // Cl_1(π/2) = -ln(√2) ≈ -0.34657
    println!("Cl_1(π/2) = {:.6} (expect ≈ -0.34657)", clausen(1, std::f64::consts::PI / 2.0));
}


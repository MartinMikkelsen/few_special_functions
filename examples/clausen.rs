use few_special_functions::clausen::clausen;
use std::f64::consts::PI;

fn main() {
    println!("Cl₂(θ) — the Clausen function of order 2");
    println!("{:>10}  {:>16}", "θ/π", "Cl₂(θ)");
    println!("{}", "-".repeat(28));

    for i in 0..=12 {
        let theta = i as f64 * PI / 6.0;
        let val = clausen(2, theta);
        println!("{:>10.4}  {:>16.12}", i as f64 / 6.0, val);
    }

    println!();
    println!(
        "Cl₂(π/2) = Catalan's constant G ≈ {:.15}",
        clausen(2, PI / 2.0)
    );
    println!("Cl₂(π/3) ≈ {:.15}", clausen(2, PI / 3.0));
    println!("Cl₂(π)   = 0 (exact): {}", clausen(2, PI));
    println!();
    println!("Cl₃(0)   = ζ(3) ≈ {:.15}", clausen(3, 0.0));
    println!("Cl₅(0)   = ζ(5) ≈ {:.15}", clausen(5, 0.0));
}

use few_special_functions::fresnel::fresnel;
use std::f64::consts::PI;

fn main() {
    println!("{:>8}  {:>14}  {:>14}", "x", "C(x)", "S(x)");
    println!("{}", "-".repeat(40));

    for i in 0..=20 {
        let x = i as f64 * 0.5;
        let (c, s, _) = fresnel(x);
        println!("{x:>8.2}  {c:>14.10}  {s:>14.10}");
    }

    println!();
    println!("Limits as x → ∞: C → 0.5, S → 0.5");
    println!("Odd functions: C(−x) = −C(x), S(−x) = −S(x)");
    println!("At x = 0: C = S = 0");
    println!("Small-x: C(x) ≈ x,  S(x) ≈ πx³/6 ≈ {:.6}·x³", PI / 6.0);
}

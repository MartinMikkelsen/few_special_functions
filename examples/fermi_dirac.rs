use few_special_functions::fermi_dirac::fermi_dirac_integral;

fn main() {
    let orders = [-0.5, 0.0, 0.5, 1.5, 2.5];

    print!("{:>6}", "x");
    for &j in &orders {
        print!("  {:>14}", format!("F_{j}(x)"));
    }
    println!();
    println!("{}", "-".repeat(6 + orders.len() * 16));

    for i in -4..=8 {
        let x = i as f64;
        print!("{x:>6.1}");
        for &j in &orders {
            print!("  {:>14.8}", fermi_dirac_integral(j, x));
        }
        println!();
    }

    println!();
    println!("F_0(x) = ln(1 + eˣ)  — exact formula");
    println!("F_j(0) = (1 − 2^(−j)) · Γ(j+1) · ζ(j+1)  for j > −1");
}

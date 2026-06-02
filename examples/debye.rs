use few_special_functions::debye::debye_function;

fn main() {
    println!("Debye function D_n(β=1, x)");
    println!("{:>6}  {:>12}  {:>12}  {:>12}  {:>12}", "x", "n=1", "n=2", "n=3", "n=4");
    println!("{}", "-".repeat(58));

    let xs = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    for &x in &xs {
        print!("{x:>6.1}");
        for n in 1..=4 {
            print!("  {:>12.8}", debye_function(n as f64, 1.0, x));
        }
        println!();
    }

    println!();
    println!("D_n(β, 0) = 1  for all n, β");
    println!("D_n(β, x) → 0  as x → ∞");
    println!();
    println!("Effect of β (D_2(β, x=2.0)):");
    for &beta in &[0.5_f64, 1.0, 2.0, 5.0] {
        println!("  β = {beta:.1}: {:.10}", debye_function(2.0, beta, 2.0));
    }
}

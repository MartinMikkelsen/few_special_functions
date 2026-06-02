/// CLI: evaluate a special function at given arguments.
///
/// Usage:
///   cargo run --bin evaluate -- clausen <n> <theta>
///   cargo run --bin evaluate -- fermi_dirac <j> <x>
///   cargo run --bin evaluate -- debye <n> <beta> <x>
///   cargo run --bin evaluate -- fresnel <x>
use few_special_functions::{
    clausen::clausen, debye::debye_function, fermi_dirac::fermi_dirac_integral, fresnel::fresnel,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.as_slice() {
        [func, rest @ ..] if func == "clausen" => {
            let n: usize = rest[0].parse().expect("n must be an integer 1..6");
            let theta: f64 = rest[1].parse().expect("theta must be a float");
            println!("Cl_{n}({theta}) = {}", clausen(n, theta));
        }
        [func, rest @ ..] if func == "fermi_dirac" => {
            let j: f64 = rest[0].parse().expect("j must be a float");
            let x: f64 = rest[1].parse().expect("x must be a float");
            println!("F_{j}({x}) = {}", fermi_dirac_integral(j, x));
        }
        [func, rest @ ..] if func == "debye" => {
            let n: f64 = rest[0].parse().expect("n must be a float");
            let beta: f64 = rest[1].parse().expect("beta must be a float");
            let x: f64 = rest[2].parse().expect("x must be a float");
            println!("D_{n}(beta={beta}, x={x}) = {}", debye_function(n, beta, x));
        }
        [func, rest @ ..] if func == "fresnel" => {
            let x: f64 = rest[0].parse().expect("x must be a float");
            let (c, s, _) = fresnel(x);
            println!("C({x}) = {c}");
            println!("S({x}) = {s}");
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  evaluate clausen <n> <theta>");
            eprintln!("  evaluate fermi_dirac <j> <x>");
            eprintln!("  evaluate debye <n> <beta> <x>");
            eprintln!("  evaluate fresnel <x>");
            std::process::exit(1);
        }
    }
}

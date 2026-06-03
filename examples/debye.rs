use few_special_functions::debye::debye_function;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Debye function D_n(β=1, x)");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "n=1", "n=2", "n=3", "n=4"
    );
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

    let orders: &[(usize, &str, RGBColor)] = &[
        (1, "D₁(x)", RED),
        (2, "D₂(x)", BLUE),
        (3, "D₃(x)", GREEN),
        (4, "D₄(x)", full_palette::ORANGE),
    ];

    let root = SVGBackend::new("debye.svg", (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Debye Functions D_n(β=1, x)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0f64..10f64, 0f64..1.05f64)?;
    chart.configure_mesh().x_desc("x").y_desc("D_n(x)").draw()?;

    let pts: Vec<f64> = (0..=500).map(|i| i as f64 * 10.0 / 500.0).collect();

    for &(n, label, color) in orders {
        chart
            .draw_series(LineSeries::new(
                pts.iter().map(|&x| (x, debye_function(n as f64, 1.0, x))),
                color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE)
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("\nPlot saved to debye.svg");
    Ok(())
}

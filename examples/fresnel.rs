use few_special_functions::fresnel::{fresnel, fresnel_c, fresnel_s};
use plotters::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let root = SVGBackend::new("fresnel.svg", (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Fresnel Integrals", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..5f64, -0.1f64..1.0f64)?;
    chart.configure_mesh().x_desc("x").draw()?;

    let pts: Vec<_> = (0..=500).map(|i| i as f64 * 5.0 / 500.0).collect();

    chart
        .draw_series(LineSeries::new(
            pts.iter().map(|&x| (x, fresnel_c(x))),
            &BLUE,
        ))?
        .label("C(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(
            pts.iter().map(|&x| (x, fresnel_s(x))),
            &RED,
        ))?
        .label("S(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE)
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("\nPlot saved to fresnel.svg");
    Ok(())
}

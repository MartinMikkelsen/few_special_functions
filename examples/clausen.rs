use few_special_functions::clausen::clausen;
use plotters::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Plot orders 1–4; Cl_1 diverges at 0 and 2π so start slightly inside
    let orders: &[(usize, &str, RGBColor)] = &[
        (1, "Cl₁(θ)", RED),
        (2, "Cl₂(θ)", BLUE),
        (3, "Cl₃(θ)", GREEN),
        (4, "Cl₄(θ)", full_palette::ORANGE),
    ];

    let root = SVGBackend::new("clausen.svg", (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Clausen Functions Cl_n(θ)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0f64..(2.0 * PI), -1.5f64..1.5f64)?;
    chart
        .configure_mesh()
        .x_desc("θ")
        .x_label_formatter(&|&v| format!("{:.2}π", v / PI))
        .draw()?;

    // Skip the first few points for n=1 to avoid the singularity at 0
    let pts: Vec<f64> = (1..=600).map(|i| i as f64 * 2.0 * PI / 600.0).collect();

    for &(n, label, color) in orders {
        chart
            .draw_series(LineSeries::new(
                pts.iter().filter_map(|&t| {
                    let v = clausen(n, t);
                    if v.abs() <= 1.5 { Some((t, v)) } else { None }
                }),
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
    println!("\nPlot saved to clausen.svg");
    Ok(())
}

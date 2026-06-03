use few_special_functions::fermi_dirac::{fermi_dirac_integral, fermi_dirac_integral_norm};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Normalized integrals F̃_j(x) = F_j(x)/Γ(j+1) — stay on a comparable scale
    let norm_orders: &[(f64, &str, RGBColor)] = &[
        (-0.5, "j = -1/2", RED),
        (0.5, "j = 1/2", BLUE),
        (1.5, "j = 3/2", GREEN),
        (2.5, "j = 5/2", full_palette::PURPLE),
    ];

    let root = SVGBackend::new("fermi_dirac.svg", (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Normalized Fermi-Dirac Integrals F̃_j(x)",
            ("sans-serif", 22),
        )
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(-4f64..8f64, 0f64..12f64)?;
    chart.configure_mesh().x_desc("x").y_desc("F̃_j(x)").draw()?;

    let pts: Vec<f64> = (0..=600).map(|i| -4.0 + i as f64 * 12.0 / 600.0).collect();

    for &(j, label, color) in norm_orders {
        chart
            .draw_series(LineSeries::new(
                pts.iter().map(|&x| (x, fermi_dirac_integral_norm(j, x))),
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
    println!("\nPlot saved to fermi_dirac.svg");
    Ok(())
}

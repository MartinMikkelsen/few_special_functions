/// Print a labelled section header.
pub fn header(title: &str) {
    println!("\n── {title} ──");
}

/// Print a row of f64 values with a fixed-width label.
pub fn row(label: &str, values: &[f64]) {
    print!("  {label:<20}");
    for v in values {
        print!("  {v:>14.8}");
    }
    println!();
}

/// Print a column-header line.
pub fn col_headers(headers: &[&str]) {
    print!("  {:<20}", "");
    for h in headers {
        print!("  {h:>14}");
    }
    println!();
    println!("  {}", "-".repeat(20 + headers.len() * 16));
}

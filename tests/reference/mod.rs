//! Reader and error measure for the mpmath tables in `tests/reference/`.
//!
//! Each table is produced by `scripts/generate_reference.py`; its header names
//! the definition, the generator and the mpmath/SciPy versions.

use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub struct Table {
    pub name: &'static str,
    columns: HashMap<String, usize>,
    rows: Vec<Vec<f64>>,
}

pub struct Row<'a> {
    table: &'a Table,
    values: &'a [f64],
}

impl Row<'_> {
    pub fn get(&self, column: &str) -> f64 {
        let index = *self
            .table
            .columns
            .get(column)
            .unwrap_or_else(|| panic!("{}: no column `{column}`", self.table.name));
        self.values[index]
    }

    /// The complex value stored in columns `{prefix}_re` and `{prefix}_im`.
    pub fn complex(&self, prefix: &str) -> Complex64 {
        Complex64::new(
            self.get(&format!("{prefix}_re")),
            self.get(&format!("{prefix}_im")),
        )
    }

    fn describe(&self, inputs: &[&str]) -> String {
        let mut text = String::new();
        for name in inputs {
            write!(text, "{name}={:e} ", self.get(name)).unwrap();
        }
        text
    }
}

impl Table {
    pub fn parse(name: &'static str, text: &str) -> Self {
        let mut lines = text.lines().filter(|line| !line.starts_with('#'));
        let header = lines.next().expect("table has a header line");
        let columns = header
            .split('\t')
            .enumerate()
            .map(|(index, column)| (column.to_string(), index))
            .collect::<HashMap<_, _>>();
        let rows = lines
            .map(|line| {
                let row = line
                    .split('\t')
                    .map(|field| {
                        field
                            .parse::<f64>()
                            .unwrap_or_else(|_| panic!("{name}: bad number `{field}`"))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(row.len(), columns.len(), "{name}: ragged row `{line}`");
                row
            })
            .collect::<Vec<_>>();
        assert!(!rows.is_empty(), "{name}: no rows");
        Self {
            name,
            columns,
            rows,
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = Row<'_>> {
        self.rows.iter().map(|values| Row {
            table: self,
            values,
        })
    }
}

/// Relative error `|actual - expected| / |expected|`, measured against the
/// modulus for complex values. At an exact zero of the reference it is the
/// absolute error `|actual|`. NaN counts as infinitely wrong.
pub fn error(actual: Complex64, expected: Complex64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    let err = if expected == Complex64::new(0.0, 0.0) {
        actual.norm()
    } else {
        (actual - expected).norm() / expected.norm()
    };
    if err.is_nan() { f64::INFINITY } else { err }
}

/// Failures collected over several checks, asserted together by [`Report::finish`]
/// so that one run prints the error summary of every check.
#[derive(Default)]
pub struct Report {
    failures: Vec<String>,
}

impl Report {
    pub fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} check(s) failed:\n{}",
            self.failures.len(),
            self.failures.join("\n\n")
        );
    }
}

/// Selects the rows a [`Check`] covers.
pub type Domain = fn(&Row) -> bool;

/// One quantity of a table checked against a stated tolerance.
pub struct Check<'a> {
    table: &'a Table,
    quantity: &'static str,
    inputs: &'static [&'static str],
    expected: &'static str,
    tolerance: f64,
    parity: Option<&'static str>,
    domain: Option<(&'static str, Domain)>,
}

impl<'a> Check<'a> {
    /// Compares `quantity` against column `expected` (or the complex pair
    /// `expected_re`, `expected_im` for [`Check::run_complex`]); `inputs` are
    /// the columns shown for failing points.
    pub fn new(
        table: &'a Table,
        quantity: &'static str,
        inputs: &'static [&'static str],
        expected: &'static str,
        tolerance: f64,
    ) -> Self {
        Self {
            table,
            quantity,
            inputs,
            expected,
            tolerance,
            parity: None,
            domain: None,
        }
    }

    /// Also report (not assert) the error of the SciPy column `column`.
    pub fn parity(mut self, column: &'static str) -> Self {
        self.parity = Some(column);
        self
    }

    /// Restrict the check to the rows where `keep` holds; `label` names that
    /// domain in the report.
    pub fn domain(mut self, label: &'static str, keep: Domain) -> Self {
        self.domain = Some((label, keep));
        self
    }

    /// Evaluates `compute` at every row, prints the error summary, and records
    /// a failure in `report` if any point exceeds the tolerance or panics.
    pub fn run(&self, report: &mut Report, compute: impl Fn(&Row) -> f64) {
        self.evaluate(
            report,
            |row| row.get(self.expected).into(),
            |row| compute(row).into(),
        );
    }

    /// Like [`Check::run`] for a complex quantity; the error is measured
    /// relative to the modulus of the reference value.
    pub fn run_complex(&self, report: &mut Report, compute: impl Fn(&Row) -> Complex64) {
        self.evaluate(report, |row| row.complex(self.expected), compute);
    }

    fn evaluate(
        &self,
        report: &mut Report,
        expected_of: impl Fn(&Row) -> Complex64,
        compute: impl Fn(&Row) -> Complex64,
    ) {
        let mut worst = (0.0_f64, String::new());
        let mut scipy_worst = 0.0_f64;
        let mut failures = Vec::new();
        let mut errors = Vec::new();
        for row in self.table.rows() {
            if let Some((_, keep)) = self.domain
                && !keep(&row)
            {
                continue;
            }
            let expected = expected_of(&row);
            if expected.re.is_nan() {
                continue;
            }
            let where_ = row.describe(self.inputs);
            match catch_unwind(AssertUnwindSafe(|| compute(&row))) {
                Ok(actual) => {
                    let err = error(actual, expected);
                    errors.push(err);
                    if err > worst.0 || worst.1.is_empty() {
                        worst = (err, where_.clone());
                    }
                    if err > self.tolerance {
                        failures.push(format!(
                            "{where_}: expected {expected:.17e}, got {actual:.17e} (error {err:.2e})"
                        ));
                    }
                }
                Err(_) => {
                    errors.push(f64::INFINITY);
                    worst = (f64::INFINITY, where_.clone());
                    failures.push(format!("{where_}: panicked"));
                }
            }
            if let Some(column) = self.parity {
                scipy_worst = scipy_worst.max(error(row.get(column).into(), expected));
            }
        }
        assert!(
            !errors.is_empty(),
            "{} {}: no rows checked",
            self.table.name,
            self.quantity
        );
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2];
        let label = match self.domain {
            Some((label, _)) => format!("{} [{label}]", self.quantity),
            None => self.quantity.to_string(),
        };
        let scipy = self
            .parity
            .map(|_| format!(", scipy max {scipy_worst:.1e}"))
            .unwrap_or_default();
        println!(
            "{:<22} {:<34} n={:<4} max {:.1e} median {:.1e} tol {:.0e}{scipy}  worst at {}",
            self.table.name,
            label,
            errors.len(),
            worst.0,
            median,
            self.tolerance,
            worst.1
        );
        if !failures.is_empty() {
            const SHOWN: usize = 8;
            let total = failures.len();
            failures.truncate(SHOWN);
            let more = if total > SHOWN {
                format!("\n... and {} more", total - SHOWN)
            } else {
                String::new()
            };
            report.failures.push(format!(
                "{} {label}: {total} of {} points outside tolerance {:e}:\n{}{more}",
                self.table.name,
                errors.len(),
                self.tolerance,
                failures.join("\n"),
            ));
        }
    }
}

use criterion::{Criterion, black_box};
use few_special_functions::fermi_dirac::fermi_dirac_integral;

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("fermi_dirac");
    g.bench_function("F_{1/2}(1.0)", |b| {
        b.iter(|| fermi_dirac_integral(black_box(0.5), black_box(1.0)))
    });
    g.bench_function("F_{3/2}(2.0)", |b| {
        b.iter(|| fermi_dirac_integral(black_box(1.5), black_box(2.0)))
    });
    g.finish();
}

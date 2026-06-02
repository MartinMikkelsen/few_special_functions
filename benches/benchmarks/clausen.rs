use criterion::{Criterion, black_box};
use few_special_functions::clausen::clausen;
use std::f64::consts::FRAC_PI_3;

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("clausen");
    g.bench_function("Cl2(pi/3)", |b| {
        b.iter(|| clausen(black_box(2), black_box(FRAC_PI_3)))
    });
    g.bench_function("Cl4(1.5)", |b| {
        b.iter(|| clausen(black_box(4), black_box(1.5)))
    });
    g.bench_function("Cl6(2.8)", |b| {
        b.iter(|| clausen(black_box(6), black_box(2.8)))
    });
    g.finish();
}

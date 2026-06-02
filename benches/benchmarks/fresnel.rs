use criterion::{Criterion, black_box};
use few_special_functions::fresnel::fresnel;

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("fresnel");
    g.bench_function("fresnel(1.0) series", |b| {
        b.iter(|| fresnel(black_box(1.0)))
    });
    g.bench_function("fresnel(5.0) asymptotic", |b| {
        b.iter(|| fresnel(black_box(5.0)))
    });
    g.finish();
}

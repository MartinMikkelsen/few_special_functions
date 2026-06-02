use criterion::{Criterion, black_box};
use few_special_functions::marcum_q::marcum_q;

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("marcum_q");
    g.bench_function("Q_1(0.2, 0.6)", |b| {
        b.iter(|| marcum_q(black_box(1.0), black_box(0.2), black_box(0.6)))
    });
    g.bench_function("Q_5(1.46, 4.0)", |b| {
        b.iter(|| marcum_q(black_box(5.0), black_box(1.46), black_box(4.0)))
    });
    g.finish();
}

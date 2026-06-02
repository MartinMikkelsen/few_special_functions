use criterion::{Criterion, black_box};
use few_special_functions::debye::debye_function;

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("debye");
    g.bench_function("D_3(beta=1, x=2)", |b| {
        b.iter(|| debye_function(black_box(3.0), black_box(1.0), black_box(2.0)))
    });
    g.bench_function("D_9(beta=1, x=3.4)", |b| {
        b.iter(|| debye_function(black_box(9.0), black_box(1.0), black_box(3.4)))
    });
    g.finish();
}

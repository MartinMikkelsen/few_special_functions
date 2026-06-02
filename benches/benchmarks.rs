use criterion::{Criterion, black_box, criterion_group, criterion_main};
use few_special_functions::{
    clausen::clausen, debye::debye_function, fermi_dirac::fermi_dirac_integral, fresnel::fresnel,
};

fn bench_clausen(c: &mut Criterion) {
    let mut group = c.benchmark_group("clausen");
    group.bench_function("Cl2(pi/3)", |b| {
        b.iter(|| clausen(black_box(2), black_box(std::f64::consts::FRAC_PI_3)))
    });
    group.bench_function("Cl4(1.5)", |b| {
        b.iter(|| clausen(black_box(4), black_box(1.5)))
    });
    group.bench_function("Cl6(2.8)", |b| {
        b.iter(|| clausen(black_box(6), black_box(2.8)))
    });
    group.finish();
}

fn bench_fermi_dirac(c: &mut Criterion) {
    let mut group = c.benchmark_group("fermi_dirac");
    group.bench_function("F_{1/2}(1.0)", |b| {
        b.iter(|| fermi_dirac_integral(black_box(0.5), black_box(1.0)))
    });
    group.bench_function("F_{3/2}(2.0)", |b| {
        b.iter(|| fermi_dirac_integral(black_box(1.5), black_box(2.0)))
    });
    group.finish();
}

fn bench_debye(c: &mut Criterion) {
    let mut group = c.benchmark_group("debye");
    group.bench_function("D_3(beta=1, x=2)", |b| {
        b.iter(|| debye_function(black_box(3.0), black_box(1.0), black_box(2.0)))
    });
    group.bench_function("D_9(beta=1, x=3.4)", |b| {
        b.iter(|| debye_function(black_box(9.0), black_box(1.0), black_box(3.4)))
    });
    group.finish();
}

fn bench_fresnel(c: &mut Criterion) {
    let mut group = c.benchmark_group("fresnel");
    group.bench_function("fresnel(1.0) series", |b| {
        b.iter(|| fresnel(black_box(1.0)))
    });
    group.bench_function("fresnel(5.0) asymptotic", |b| {
        b.iter(|| fresnel(black_box(5.0)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_clausen,
    bench_fermi_dirac,
    bench_debye,
    bench_fresnel
);
criterion_main!(benches);

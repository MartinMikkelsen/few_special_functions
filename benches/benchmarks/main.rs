mod clausen;
mod debye;
mod fermi_dirac;
mod fresnel;
mod marcum_q;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    clausen::bench,
    fermi_dirac::bench,
    debye::bench,
    fresnel::bench,
    marcum_q::bench
);
criterion_main!(benches);

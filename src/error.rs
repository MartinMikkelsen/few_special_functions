//! Numerical failures reported by the `*_checked` functions.

/// A numerical failure of a `*_checked` function.
///
/// Arguments outside a function's domain are not errors: the checked and the
/// plain function both return NaN for them, as documented on each function.
/// An error means the algorithm itself failed at an argument it covers, and
/// the plain function returns NaN in that case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpecialError {
    /// An iterative method reached its iteration cap before meeting its
    /// tolerance. Each module documents its caps and tolerances.
    #[error("{family}: no convergence within {max_iterations} iterations")]
    NoConvergence {
        /// The function family, e.g. `"dawson"`.
        family: &'static str,
        /// The cap that was reached.
        max_iterations: usize,
    },
}

impl SpecialError {
    pub(crate) const fn no_convergence(family: &'static str, max_iterations: usize) -> Self {
        Self::NoConvergence {
            family,
            max_iterations,
        }
    }
}

/// The iteration cap of a loop whose documented cap is `default`.
///
/// Unit tests can lower every cap on the current thread with
/// `with_iteration_cap`; outside tests this is `default`.
pub(crate) fn iteration_cap(default: usize) -> usize {
    #[cfg(test)]
    if let Some(cap) = test_cap::CAP.get() {
        return cap.min(default);
    }
    default
}

/// Runs `f` with every iteration cap on this thread lowered to at most `cap`.
#[cfg(test)]
pub(crate) fn with_iteration_cap<T>(cap: usize, f: impl FnOnce() -> T) -> T {
    let previous = test_cap::CAP.replace(Some(cap));
    let result = f();
    test_cap::CAP.set(previous);
    result
}

#[cfg(test)]
mod test_cap {
    use std::cell::Cell;

    thread_local! {
        pub static CAP: Cell<Option<usize>> = const { Cell::new(None) };
    }
}

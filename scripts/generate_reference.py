# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "scipy==1.18.1"]
# ///
"""Generate the mpmath reference tables in tests/reference/.

Run from the repository root:

    uv run scripts/generate_reference.py            # every table
    uv run scripts/generate_reference.py coulomb    # one family

Every table is computed with mpmath at `DPS` significant digits and written
with `DIGITS` significant digits, so the stored value is the correctly rounded
reference to well beyond f64 precision. Inputs are Python floats, so they are
exactly representable in f64 and the Rust tests evaluate at the same points.
Where SciPy implements the same function, its f64 value is stored in a
`scipy` column for parity comparison; it is never used as the reference.

Several definitions are evaluated two independent ways (for example a
quadrature of the defining integral and a closed form); the script stops if
the two disagree, so a table is never written from a single unchecked path.
"""

from __future__ import annotations

import datetime
import pathlib
import platform
import sys

import mpmath
import scipy
import scipy.special as sc
import scipy.stats as st
from mpmath import mp, mpc, mpf

DPS = 50
DIGITS = 22
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "reference"

mp.dps = DPS
# Bits below which mpmath may report an exactly zero hypergeometric sum as 0.
ZEROPREC = 4 * mp.prec


def fmt(value) -> str:
    return mpmath.nstr(mpf(value), DIGITS, strip_zeros=False, min_fixed=0, max_fixed=0)


def fmt_float(value: float) -> str:
    return repr(float(value))


def agree(a, b, what: str, rel: float = mpf(10) ** (-(DPS - 15)),
          absolute: float = mpf(10) ** (-(DPS - 5))) -> None:
    scale = max(abs(a), abs(b))
    if abs(a - b) > rel * scale and abs(a - b) > absolute:
        raise RuntimeError(f"cross-check failed for {what}: {a} vs {b}")


def write(name: str, definition: str, columns: list[str], rows: list[list[str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = [
        f"# dataset: {name}",
        f"# definition: {definition}",
        "# generator: scripts/generate_reference.py",
        f"# mpmath {mpmath.__version__}, scipy {scipy.__version__}, "
        f"python {platform.python_version()}, generated {datetime.date.today().isoformat()}",
        f"# working precision: {DPS} digits; values written to {DIGITS} digits",
        "\t".join(columns),
    ]
    with open(OUT / f"{name}.tsv", "w") as handle:
        handle.write("\n".join(header) + "\n")
        for row in rows:
            assert len(row) == len(columns), (name, row)
            handle.write("\t".join(row) + "\n")
    print(f"{name}: {len(rows)} rows", file=sys.stderr)


def complex_cols(value) -> list[str]:
    value = mpc(value)
    return [fmt(value.real), fmt(value.imag)]


# ----------------------------------------------------------------------------
# Clausen functions


def clausen_tables() -> None:
    thetas = [1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 1.25, 2.0, 2.5, 3.0, 3.5, 4.0,
              5.0, 6.0, 6.2, 7.5, 10.0, 100.0, -0.3, -2.7]
    rows = []
    for n in range(1, 7):
        for theta in thetas:
            t = mpf(theta)
            value = mpmath.clsin(n, t) if n % 2 == 0 else mpmath.clcos(n, t)
            li = mpmath.polylog(n, mpmath.expjpi(t / mpmath.pi))
            series = li.imag if n % 2 == 0 else li.real
            agree(value, series, f"Cl_{n}({theta})", rel=mpf(10) ** -25)
            rows.append([str(n), fmt_float(theta), fmt(value)])
    write("clausen", "Cl_n(theta) = sum_k sin(k theta)/k^n (n even), cos(k theta)/k^n (n odd)",
          ["n", "theta", "value"], rows)


# ----------------------------------------------------------------------------
# Fermi-Dirac and Bose-Einstein integrals


def fermi_dirac_quad(j, x):
    f = lambda t: t**j / (mpmath.exp(t - x) + 1)
    points = [0, max(x, 0) + 1, max(x, 0) + 40, mpmath.inf]
    return mpmath.quad(f, points)


def fermi_dirac_tables() -> None:
    js = [-0.5, -0.4, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.3, 10.0]
    xs = [-50.0, -20.0, -5.0, -2.0, -1.0, -0.999, -0.5, 0.0, 0.001, 0.5, 1.0, 2.0, 5.0, 10.0,
          20.0, 50.0, 100.0, 1000.0, 10000.0]
    rows = []
    for j in js:
        for x in xs:
            jj, xx = mpf(j), mpf(x)
            closed = -mpmath.gamma(jj + 1) * mpmath.polylog(jj + 1, -mpmath.exp(xx))
            quad = fermi_dirac_quad(jj, xx)
            agree(closed, quad, f"F_{j}({x})", rel=mpf(10) ** -25)
            rows.append([fmt_float(j), fmt_float(x), fmt(quad), fmt(quad / mpmath.gamma(jj + 1))])
    write("fermi_dirac", "F_j(x) = int_0^inf t^j/(exp(t-x)+1) dt; norm = F_j(x)/Gamma(j+1)",
          ["j", "x", "value", "norm"], rows)


def bose_einstein_tables() -> None:
    ks = [-4.5, -3.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5]
    etas = [-50.0, -10.0, -3.0, -1.0, -0.5, -0.1, -0.01, -1e-4, 0.0]
    rows = []
    for k in ks:
        for eta in etas:
            if eta == 0.0 and k <= 0.0:
                continue  # Li_{k+1}(1) diverges for k + 1 <= 1
            kk, ee = mpf(k), mpf(eta)
            norm = mpmath.polylog(kk + 1, mpmath.exp(ee))
            if k > -1:
                f = lambda t: t**kk / (mpmath.exp(t - ee) - 1)
                quad = mpmath.quad(f, [0, 1, 40, mpmath.inf])
                agree(quad, mpmath.gamma(kk + 1) * norm, f"BE_{k}({eta})", rel=mpf(10) ** -25)
                value = fmt(mpmath.gamma(kk + 1) * norm)
            else:
                value = "nan"
            rows.append([fmt_float(k), fmt_float(eta), value, fmt(norm)])
    write("bose_einstein",
          "norm = Li_{k+1}(exp(eta)); value = Gamma(k+1) norm = int_0^inf t^k/(exp(t-eta)-1) dt (k > -1, else nan)",
          ["k", "eta", "value", "norm"], rows)


# ----------------------------------------------------------------------------
# Debye functions


def debye_tables() -> None:
    ns = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0]
    xs = [1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    rows = []
    for n in ns:
        for beta in betas:
            if n - beta <= -1:
                continue  # the defining integral diverges at t = 0
            for x in xs:
                nn, bb, xx = mpf(n), mpf(beta), mpf(x)
                f = lambda t: t**nn / mpmath.expm1(t) ** bb
                value = nn / xx**nn * mpmath.quad(f, [0, xx])
                rows.append([fmt_float(n), fmt_float(beta), fmt_float(x), fmt(value)])
    write("debye", "D_n(beta, x) = (n/x^n) int_0^x t^n/(exp(t)-1)^beta dt",
          ["n", "beta", "x", "value"], rows)


# ----------------------------------------------------------------------------
# Dawson, Voigt and Fresnel


def dawson_tables() -> None:
    xs = [1e-10, 1e-5, 0.01, 0.1, 0.5, 0.9, 0.924, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0,
          10.0, 20.0, 50.0, 100.0, 1e3, 1e5, 1e10]
    rows = []
    for x in xs + [-v for v in xs[::3]]:
        xx = mpf(x)
        value = mpmath.sqrt(mpmath.pi) / 2 * mpmath.exp(-xx**2) * mpmath.erfi(xx)
        if abs(x) < 10:
            quad = mpmath.exp(-xx**2) * mpmath.quad(lambda t: mpmath.exp(t**2), [0, xx])
            agree(value, quad, f"dawson({x})", rel=mpf(10) ** -25)
        rows.append([fmt_float(x), fmt(value), fmt_float(sc.dawsn(x))])
    write("dawson", "D(x) = exp(-x^2) int_0^x exp(t^2) dt", ["x", "value", "scipy"], rows)


def voigt_quad(x, y):
    """(y/pi) int exp(-t^2)/((x-t)^2 + y^2) dt, with the Lorentzian peak at t = x
    resolved by breakpoints at multiples of y."""
    points = {-mpmath.inf, mpf(-6), mpf(0), mpf(6), mpmath.inf}
    if abs(x) < 40:
        points |= {x + d for d in (-1, -100 * y, -y, 0, y, 100 * y, 1)}
    points = sorted(points)
    return y / mpmath.pi * mpmath.quad(lambda t: mpmath.exp(-t**2) / ((x - t) ** 2 + y**2), points)


def voigt_tables() -> None:
    xs = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 7.0, 10.0, 15.0, 30.0, 70.0, 100.0, 300.0, 1e3, 1e4, 1e6, -1.5]
    ys = [0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 30.0, 70.0, 100.0, 1e3]
    rows = []
    for x in xs:
        for y in ys:
            xx, yy = mpf(x), mpf(y)
            if y == 0:
                value = mpmath.exp(-xx**2)
            else:
                z = mpc(xx, yy)
                # Re w loses about log10(|w| / Re w) <= log10(1 + |x|/y) digits to cancellation.
                with mp.workdps(DPS + 20 + int(mpmath.log10(1 + abs(xx) / yy))):
                    value = (mpmath.exp(-z**2) * mpmath.erfc(-1j * z)).real
                if abs(x) <= 1e3:
                    agree(value, voigt_quad(xx, yy), f"voigt({x}, {y})", rel=mpf(10) ** -25)
            rows.append([fmt_float(x), fmt_float(y), fmt(value), fmt_float(sc.wofz(complex(x, y)).real)])
    write("voigt", "K(x, y) = (y/pi) int exp(-t^2)/((x-t)^2+y^2) dt = Re w(x + i y)",
          ["x", "y", "value", "scipy"], rows)


def fresnel_tables() -> None:
    xs = [1e-6, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 30.0, 100.0, 1e3, -0.7, -4.2]
    rows = []
    for x in xs:
        xx = mpf(x)
        c, s = mpmath.fresnelc(xx), mpmath.fresnels(xx)
        if abs(x) <= 5:
            agree(c, mpmath.quad(lambda t: mpmath.cos(mpmath.pi * t**2 / 2), [0, xx]), f"C({x})",
                  rel=mpf(10) ** -25)
        ss, cc = sc.fresnel(x)
        rows.append([fmt_float(x), fmt(c), fmt(s), fmt_float(cc), fmt_float(ss)])
    write("fresnel", "C(x) = int_0^x cos(pi t^2/2) dt, S(x) = int_0^x sin(pi t^2/2) dt",
          ["x", "c", "s", "scipy_c", "scipy_s"], rows)

    zs = [(0.5, 0.5), (1.0, 1.0), (2.0, -1.0), (-1.5, 0.3), (3.0, 0.1), (0.1, 2.0), (4.0, 2.0), (-0.2, -0.7)]
    rows = []
    for re, im in zs:
        z = mpc(re, im)
        rows.append([fmt_float(re), fmt_float(im), *complex_cols(mpmath.fresnelc(z)),
                     *complex_cols(mpmath.fresnels(z))])
    write("fresnel_complex", "C(z), S(z) on the principal branch (entire functions)",
          ["z_re", "z_im", "c_re", "c_im", "s_re", "s_im"], rows)


# ----------------------------------------------------------------------------
# Marcum Q


def marcum_q_series(mu, a, b):
    x, y = a**2 / 2, b**2 / 2
    total, k = mpf(0), 0
    while True:
        term = mpmath.exp(-x) * x**k / mpmath.factorial(k) * mpmath.gammainc(mu + k, y, regularized=True)
        total += term
        if k > x + 10 and term < total * mpf(10) ** (-DPS):
            return total
        k += 1


def marcum_q_tables() -> None:
    mus = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
    avals = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 30.0]
    bvals = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 30.0]
    rows = []
    for mu in mus:
        for a in avals:
            for b in bvals:
                m, aa, bb = mpf(mu), mpf(a), mpf(b)
                value = marcum_q_series(m, aa, bb)
                if value < mpf(10) ** -300:
                    continue
                if 0 < a <= 5 and 0 < b <= 10:
                    f = lambda t: t * (t / aa) ** (m - 1) * mpmath.exp(-(t**2 + aa**2) / 2) * mpmath.besseli(m - 1, aa * t)
                    agree(value, mpmath.quad(f, [bb, bb + 10, mpmath.inf]), f"Q_{mu}({a},{b})",
                          rel=mpf(10) ** -25)
                scipy_value = float(st.ncx2.sf(b * b, 2 * mu, a * a)) if a > 0 else float(sc.gammaincc(mu, b * b / 2))
                rows.append([fmt_float(mu), fmt_float(a), fmt_float(b), fmt(value), fmt_float(scipy_value)])
    write("marcum_q", "Q_mu(a, b) = int_b^inf t (t/a)^(mu-1) exp(-(t^2+a^2)/2) I_{mu-1}(a t) dt",
          ["mu", "a", "b", "value", "scipy"], rows)

    rows = []
    for m in [1, 2, 3, 5]:
        for a in [0.1, 1.0, 2.0, 5.0, 10.0]:
            for b in [0.1, 1.0, 2.0, 5.0, 10.0, 20.0]:
                aa, bb = mpf(a), mpf(b)
                value = -(bb**m) / aa ** (m - 1) * mpmath.exp(-(aa**2 + bb**2) / 2) * mpmath.besseli(m - 1, aa * bb)
                check = mpmath.diff(lambda t: marcum_q_series(mpf(m), aa, t), bb)
                agree(value, check, f"dQ_{m}/db({a},{b})", rel=mpf(10) ** -20)
                if abs(value) < mpf(10) ** -300:
                    continue
                rows.append([str(m), fmt_float(a), fmt_float(b), fmt(value)])
    write("marcum_q_db", "dQ_M/db = -(b^M/a^(M-1)) exp(-(a^2+b^2)/2) I_{M-1}(a b)",
          ["m", "a", "b", "value"], rows)


# ----------------------------------------------------------------------------
# Parabolic-cylinder functions


def pcf_uv(a, x):
    """U(a, x) and V(a, x) from the even and odd solutions (DLMF sections 12.4 and 12.7).

    At a = -n - 1/2 the polynomial factor of u1 or u2 has exact zeros on the
    grid (for example 1 - x^2 at a = -5/2), so the hypergeometric sums are
    told to accept a zero result. Unlike mpmath's pcfv, this form has no removable singularity at
    a = -n - 1/2, so it also covers the points where pcfv fails to converge.
    """
    # The sums cancel by up to exp(x^2/2) where a solution is recessive.
    with mp.workdps(2 * DPS + int(mpf(x) ** 2 / 4)):
        a, x = mpf(a), mpf(x)
        g = mpmath.exp(-x**2 / 4)
        u1 = g * mpmath.hyp1f1(a / 2 + mpf(1) / 4, mpf(1) / 2, x**2 / 2, zeroprec=ZEROPREC)
        u2 = x * g * mpmath.hyp1f1(a / 2 + mpf(3) / 4, mpf(3) / 2, x**2 / 2, zeroprec=ZEROPREC)
        u0 = mpmath.sqrt(mpmath.pi) * mpmath.rgamma(mpf(3) / 4 + a / 2) / 2 ** (a / 2 + mpf(1) / 4)
        du0 = -mpmath.sqrt(mpmath.pi) * mpmath.rgamma(mpf(1) / 4 + a / 2) / 2 ** (a / 2 - mpf(1) / 4)
        v0 = 2 ** (a / 2 + mpf(1) / 4) * mpmath.sinpi(mpf(3) / 4 - a / 2) * mpmath.rgamma(mpf(3) / 4 - a / 2)
        dv0 = 2 ** (a / 2 + mpf(3) / 4) * mpmath.sinpi(mpf(1) / 4 - a / 2) * mpmath.rgamma(mpf(1) / 4 - a / 2)
        return u0 * u1 + du0 * u2, v0 * u1 + dv0 * u2


def parabolic_cylinder_tables() -> None:
    avals = [-10.0, -5.0, -3.3, -2.5, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.7, 2.5, 5.0, 7.5, 10.0, 20.0]
    xvals = [-20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []
    for a in avals:
        for x in xvals:
            aa, xx = mpf(a), mpf(x)
            u, v = pcf_uv(aa, xx)
            agree(u, mpmath.pcfu(aa, xx, zeroprec=ZEROPREC), f"U({a}, {x})", rel=mpf(10) ** -25)
            try:
                agree(v, mpmath.pcfv(aa, xx, zeroprec=ZEROPREC), f"V({a}, {x})", rel=mpf(10) ** -25)
            except ValueError:
                pass  # pcfv does not converge here; the series form stands alone
            if x < 0:
                # DLMF 12.2.15 and 12.2.16 from the values at -x.
                up, vp = pcf_uv(aa, -xx)
                agree(u, -mpmath.sinpi(aa) * up + mpmath.pi * mpmath.rgamma(aa + mpf(1) / 2) * vp,
                      f"U({a}, {x}) reflection", rel=mpf(10) ** -25, absolute=0)
                agree(v, mpmath.cospi(aa) * mpmath.rgamma(mpf(1) / 2 - aa) * up + mpmath.sinpi(aa) * vp,
                      f"V({a}, {x}) reflection", rel=mpf(10) ** -25, absolute=0)
            du = mpmath.diff(lambda t: pcf_uv(aa, t)[0], xx)
            dv = mpmath.diff(lambda t: pcf_uv(aa, t)[1], xx)
            rows.append([fmt_float(a), fmt_float(x), fmt(u), fmt(du), fmt(v), fmt(dv)])
    write("parabolic_cylinder_uv", "U(a, x), V(a, x) (DLMF 12.2) and their x-derivatives",
          ["a", "x", "u", "du", "v", "dv"], rows)

    rows = []
    for a in [-3.0, -1.0, 0.0, 0.5, 1.0, 3.0]:
        for x in [-4.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]:
            aa, xx = mpf(a), mpf(x)
            w = mpmath.pcfw(aa, xx)
            dw = mpmath.diff(lambda t: mpmath.pcfw(aa, t), xx)
            if x == 0:
                ratio = mpmath.gamma(mpc(mpf(1) / 4, aa / 2)) / mpmath.gamma(mpc(mpf(3) / 4, aa / 2))
                agree(w, 2 ** mpf(-0.75) * mpmath.sqrt(abs(ratio)), f"W({a}, 0)")  # DLMF section 12.14
            mirror = mpmath.pcfw(aa, -xx)
            dmirror = mpmath.diff(lambda t: mpmath.pcfw(aa, t), -xx)
            # Wronskian W{W(a,x), W(a,-x)} = 1 (DLMF section 12.14)
            agree(w * -dmirror - dw * mirror, mpf(1), f"Wronskian W({a}, {x})", rel=mpf(10) ** -25)
            rows.append([fmt_float(a), fmt_float(x), fmt(w), fmt(dw)])
    write("parabolic_cylinder_w", "W(a, x) (DLMF 12.14) and its x-derivative",
          ["a", "x", "w", "dw"], rows)

    rows = []
    for nu in [-3.5, -1.0, 0.0, 1.0, 2.5, 5.0]:
        for x in [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            nn, xx = mpf(nu), mpf(x)
            d = pcf_uv(-nn - mpf(1) / 2, xx)[0]
            agree(d, mpmath.pcfd(nn, xx, zeroprec=ZEROPREC), f"D_{nu}({x})", rel=mpf(10) ** -25)
            dd = mpmath.diff(lambda t: pcf_uv(-nn - mpf(1) / 2, t)[0], xx)
            rows.append([fmt_float(nu), fmt_float(x), fmt(d), fmt(dd), fmt_float(sc.pbdv(nu, x)[0])])
    write("parabolic_cylinder_d", "D_nu(x) = U(-nu-1/2, x) and its x-derivative",
          ["nu", "x", "d", "dd", "scipy"], rows)


# ----------------------------------------------------------------------------
# Whittaker functions


def whittaker_w_degenerate(kappa, mu, z):
    """W for kappa - mu - 1/2 = N a non-negative integer (either sign of mu, as W is
    even in mu): (-1)^N N! exp(-z/2) z^(mu+1/2) L_N^(2 mu)(z) (DLMF section 13.18).
    Returns None for other parameters."""
    for m in (mu, -mu):
        n = kappa - m - mpf(1) / 2
        if n >= 0 and n == int(n):
            n = int(n)
            # Explicit sum: mpmath.laguerre fails to converge at the zeros.
            laguerre = mpmath.fsum(mpmath.binomial(n + 2 * m, n - j) * (-z) ** j / mpmath.factorial(j)
                                   for j in range(n + 1))
            return (-1) ** n * mpmath.factorial(n) * mpmath.exp(-z / 2) * z ** (m + mpf(1) / 2) * laguerre
    return None


def whittaker_tables() -> None:
    rows = []
    # kappa -/+ mu - 1/2 a non-negative integer, where W reduces to a Laguerre
    # polynomial, occurs for example at (kappa, mu) = (1.5, 1.0), (2.0, 0.5),
    # (3.0, 1.5), (5.0, 2.5), (1.5, 0.0) and (1.2, -1.3).
    for kappa in [-5.0, -2.0, -1.3, -0.5, 0.0, 0.5, 1.2, 1.5, 2.0, 3.0, 4.75, 5.0]:
        for mu in [-1.3, 0.0, 0.25, 0.5, 1.0, 1.5, 2.3, 2.5, 3.5]:
            for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 50.0, 100.0, 1e3]:
                k, m, zz = mpf(kappa), mpf(mu), mpf(z)
                # Degenerate parameters put exact zeros on the grid (M and W are
                # proportional to 1 - z at kappa = 3/2, mu = 0).
                wm = mpmath.whitm(k, m, zz, zeroprec=ZEROPREC)
                dm = mpmath.diff(lambda t: mpmath.whitm(k, m, t, zeroprec=ZEROPREC), zz)
                if whittaker_w_degenerate(k, m, zz) is None:
                    ww = mpmath.whitw(k, m, zz)
                    dw = mpmath.diff(lambda t: mpmath.whitw(k, m, t), zz)
                else:
                    ww = whittaker_w_degenerate(k, m, zz)
                    dw = mpmath.diff(lambda t: whittaker_w_degenerate(k, m, t), zz)
                    try:
                        agree(ww, mpmath.whitw(k, m, zz), f"W({kappa}, {mu}, {z}) Laguerre form",
                              rel=mpf(10) ** -25)
                    except ValueError:
                        pass  # whitw does not converge at the zeros of the polynomial
                # Wronskian W{M, W} = -Gamma(1+2mu)/Gamma(1/2+mu-kappa) (DLMF section 13.14)
                agree(wm * dw - dm * ww, -mpmath.gamma(1 + 2 * m) * mpmath.rgamma(mpf(1) / 2 + m - k),
                      f"Wronskian M,W({kappa}, {mu}, {z})", rel=mpf(10) ** -25)
                rows.append([fmt_float(kappa), fmt_float(mu), fmt_float(z), fmt(wm), fmt(dm), fmt(ww), fmt(dw)])
    write("whittaker", "M_{kappa,mu}(z), W_{kappa,mu}(z) (DLMF 13.14) and their z-derivatives",
          ["kappa", "mu", "z", "m", "dm", "w", "dw"], rows)

    rows = []
    cases = [((0.5, 0.2), (0.75, -0.1), (1.0, 2.0)), ((-1.0, 0.0), (1.5, 0.0), (2.0, -1.0)),
             ((0.0, 1.0), (0.5, 0.5), (0.5, 0.5)), ((2.0, -0.5), (0.3, 0.0), (5.0, 3.0)),
             ((0.0, 0.0), (0.5, 0.0), (-1.0, 1.0)), ((1.0, 0.0), (1.0, 0.0), (3.0, 0.0))]
    for kappa, mu, z in cases:
        k, m, zz = mpc(*kappa), mpc(*mu), mpc(*z)
        rows.append([*map(fmt_float, (*kappa, *mu, *z)),
                     *complex_cols(mpmath.whitm(k, m, zz)), *complex_cols(mpmath.whitw(k, m, zz))])
    write("whittaker_complex", "M_{kappa,mu}(z), W_{kappa,mu}(z) on the principal branch",
          ["kappa_re", "kappa_im", "mu_re", "mu_im", "z_re", "z_im", "m_re", "m_im", "w_re", "w_im"], rows)


# ----------------------------------------------------------------------------
# Coulomb wave functions


def coulomb_tables() -> None:
    rows = []
    for ell in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        for eta in [-50.0, -20.0, -5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            for rho in [1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
                l, e, r = mpf(ell), mpf(eta), mpf(rho)
                f = mpmath.coulombf(l, e, r)
                g = mpmath.coulombg(l, e, r)
                if abs(g) > mpf(10) ** 300 or abs(f) < mpf(10) ** -300:
                    continue  # outside the f64 range
                df = mpmath.diff(lambda t: mpmath.coulombf(l, e, t), r)
                dg = mpmath.diff(lambda t: mpmath.coulombg(l, e, t), r)
                # Wronskian F'G - FG' = 1 (DLMF section 33.2)
                agree(df * g - f * dg, mpf(1), f"Wronskian F,G({ell}, {eta}, {rho})", rel=mpf(10) ** -25)
                rows.append([fmt_float(ell), fmt_float(eta), fmt_float(rho), fmt(f), fmt(g)])
    write("coulomb", "F_l(eta, rho), G_l(eta, rho) (DLMF 33.2); H+- = G +- i F",
          ["ell", "eta", "rho", "f", "g"], rows)

    rows = []
    for ell in [0.0, 1.0, 2.0, 5.0, 0.5]:
        for eta in [-5.0, -0.5, 0.0, 0.5, 2.0, 10.0]:
            rows.append([fmt_float(ell), fmt_float(eta), fmt(mpmath.coulombc(mpf(ell), mpf(eta)))])
    write("coulomb_normalization", "C_l(eta) = 2^l exp(-pi eta/2) |Gamma(l+1+i eta)| / Gamma(2l+2)",
          ["ell", "eta", "value"], rows)


def coulomb_phi(ell, eta, rho):
    """phi_l(eta, rho) = (2 eta rho)^(l+1) e^(i rho) M(l+1+i eta, 2l+2, -2 i rho) / Gamma(2l+2).

    The regularized Kummer function is summed directly, so negative integer
    2l+2 (reached by l -> -l-1) needs no limit.
    """
    a, b, z = ell + 1 + 1j * eta, 2 * ell + 2, -2j * rho
    total, term, k = mpc(0), mpc(1), 0
    while True:
        contribution = term * mpmath.rgamma(b + k)
        total += contribution
        if k > abs(z) + 10 and abs(contribution) < abs(total) * mpf(10) ** (-DPS):
            break
        term *= (a + k) * z / (k + 1)
        k += 1
    return (2 * eta * rho) ** (ell + 1) * mpmath.expj(rho) * total


def coulomb_phi_dot(ell, eta, rho):
    """d phi_l / d l, from the series of `coulomb_phi` differentiated term by term.

    With c_k = (a)_k z^k / k!, a = l+1+i eta and b = 2l+2, the term
    c_k / Gamma(b+k) has l-derivative (dc_k/da) / Gamma(b+k) + 2 c_k (1/Gamma)'(b+k),
    where (1/Gamma)'(x) = -psi(x)/Gamma(x), and (-1)^n n! at x = -n.
    """
    a, b, z = ell + 1 + 1j * eta, 2 * ell + 2, -2j * rho

    def rgamma_prime(x):
        if x == mpmath.floor(x) and x <= 0:
            n = int(-x)
            return (-1) ** n * mpmath.factorial(n)
        return -mpmath.digamma(x) * mpmath.rgamma(x)

    value, derivative = mpc(0), mpc(0)
    term, term_d_a, k = mpc(1), mpc(0), 0
    while True:
        r, r_prime = mpmath.rgamma(b + k), rgamma_prime(b + k)
        value += term * r
        contribution = term_d_a * r + 2 * term * r_prime
        derivative += contribution
        if k > abs(z) + 10 and abs(contribution) < abs(derivative) * mpf(10) ** (-DPS) \
                and abs(term * r) < abs(value) * mpf(10) ** (-DPS):
            break
        term_d_a = (term_d_a * (a + k) + term) * z / (k + 1)
        term *= (a + k) * z / (k + 1)
        k += 1
    base = 2 * eta * rho
    return base ** (ell + 1) * mpmath.expj(rho) * (mpmath.log(base) * value + derivative)


def coulomb_psi_i(ell, eta, rho):
    """(psi, i) at integer or half-integer l, with d phi/dl evaluated two ways."""
    def dphi(order):
        numeric = mpmath.diff(lambda t: coulomb_phi(t, eta, rho), order)
        analytic = coulomb_phi_dot(order, eta, rho)
        agree(numeric, analytic, f"dphi/dl at ({order}, {eta}, {rho})", rel=mpf(10) ** -25)
        return analytic

    start = 0 if ell == int(ell) else mpf(1) / 2
    gamow = mpmath.fprod(1 + (start + j) ** 2 / eta**2 for j in range(int(ell - start) + 1))
    psi = gamow * dphi(ell) / 2 + dphi(-ell - 1) / 2
    # C_l(eta) Gamma(2l+2) written with log-gamma, as in the complex Rust code.
    normalization_gamma = mpmath.exp(ell * mpmath.log(2) - mpmath.pi * eta / 2
                                     + (mpmath.loggamma(ell + 1 + 1j * eta)
                                        + mpmath.loggamma(ell + 1 - 1j * eta)) / 2)
    return psi, normalization_gamma / (2 * eta) ** (ell + 1) * psi


def coulomb_auxiliary_tables() -> None:
    rows = []
    points = [(1.0, 0.5, 2.0), (1.0, 0.2, 1.5), (2.0, 1.2, 3.0), (0.0, 2.0, 0.7)]
    points += [(ell, eta, rho) for ell in (0.0, 1.0, 2.0, 4.0, 6.0)
               for eta in (0.5, -0.7, 2.0) for rho in (0.5, 2.0, 5.0)]
    points += [(ell, eta, rho) for ell in (0.5, 1.5, 3.5) for eta in (0.5, -0.7)
               for rho in (1.0, 4.0)]
    for ell, eta, rho in points:
        l, e, r = mpf(ell), mpf(eta), mpf(rho)
        phase = r - l * mpmath.pi / 2 - e * mpmath.log(2 * r) + mpmath.loggamma(mpc(l + 1, e)).imag
        aux_g = mpmath.digamma(mpc(l + 1, e)).real - mpmath.log(abs(e))
        psi, i_value = coulomb_psi_i(l, e, r)
        rows.append([fmt_float(ell), fmt_float(eta), fmt_float(rho), fmt(phase), fmt(aux_g),
                     *complex_cols(coulomb_phi_dot(l, e, r)),
                     *complex_cols(psi), *complex_cols(i_value)])
    write("coulomb_auxiliary",
          "phase = rho - l pi/2 - eta ln(2 rho) + arg Gamma(l+1+i eta); "
          "aux_g = Re psi(l+1+i eta) - ln|eta|; "
          "phi_dot = d phi_l/dl with phi_l = (2 eta rho)^(l+1) e^(i rho) M(l+1+i eta, 2l+2, -2 i rho) / Gamma(2l+2); "
          "psi = w_l(eta) phi_dot_l / 2 + phi_dot_{-l-1} / 2 with w_l = prod_j (1 + j^2/eta^2), "
          "j = 0, 1, ..., l or 1/2, 3/2, ..., l; "
          "i = C_l(eta) Gamma(2l+2) / (2 eta)^(l+1) psi",
          ["ell", "eta", "rho", "phase", "aux_g", "phi_dot_re", "phi_dot_im",
           "psi_re", "psi_im", "i_re", "i_im"], rows)

    rows = []
    for ell in (1.0, 2.0, 1.5):
        for eta in (complex(0.5, 0.2), complex(-0.7, 0.1)):
            for rho in (complex(1.2, 0.1), complex(2.0, -0.3)):
                e, r = mpc(eta), mpc(rho)
                psi, i_value = coulomb_psi_i(mpf(ell), e, r)
                rows.append([fmt_float(ell), fmt_float(eta.real), fmt_float(eta.imag),
                             fmt_float(rho.real), fmt_float(rho.imag),
                             *complex_cols(psi), *complex_cols(i_value)])
    write("coulomb_auxiliary_complex",
          "psi and i of coulomb_auxiliary at complex eta and rho; "
          "C_l(eta) Gamma(2l+2) = exp(l ln 2 - pi eta/2 + (lnGamma(l+1+i eta) + lnGamma(l+1-i eta))/2)",
          ["ell", "eta_re", "eta_im", "rho_re", "rho_im", "psi_re", "psi_im", "i_re", "i_im"], rows)


TABLES = {
    "clausen": clausen_tables,
    "fermi_dirac": fermi_dirac_tables,
    "bose_einstein": bose_einstein_tables,
    "debye": debye_tables,
    "dawson": dawson_tables,
    "voigt": voigt_tables,
    "fresnel": fresnel_tables,
    "marcum_q": marcum_q_tables,
    "parabolic_cylinder": parabolic_cylinder_tables,
    "whittaker": whittaker_tables,
    "coulomb": coulomb_tables,
    "coulomb_auxiliary": coulomb_auxiliary_tables,
}


def main() -> None:
    """Generate the tables named on the command line, or all of them."""
    names = sys.argv[1:] or list(TABLES)
    for name in names:
        TABLES[name]()


if __name__ == "__main__":
    main()

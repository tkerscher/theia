import pytest

import numpy as np
import hephaistos as hp

from ctypes import Structure, c_float
from hephaistos.glsl import vec2
from numpy.lib.recfunctions import structured_to_unstructured

from theia.compiler import compileShader
from theia.device import getEnabledFma

FLT_MIN = float(np.finfo(np.float32).tiny)
FLT_MAX = float(np.finfo(np.float32).max)
EPS = 2.0**-23  # 1 ULP

# special expectations in the tables of special values
POVF = "+overflow"  # +inf or FLT_MAX, Vulkan allows both
NOVF = "-overflow"  # -inf or -FLT_MAX
TINY = "tiny"  # zero or (positive) subnormal, as subnormals may be flushed

# some math routines tested here rely on correctly fused fma to achieve their
# guaranteed error bounds. If the device does not support it, we cannot test it
requiresFma = pytest.mark.skipif(
    not getEnabledFma().float32,
    reason="accuracy test requires support for guaranteed fused fma",
)

#################################### HELPER ####################################


def f32(x):
    return np.asarray(x, dtype=np.float32)


def f64(x):
    return np.asarray(x, dtype=np.float64)


def c128(v):
    """structured vec2 array -> complex128"""
    out = np.empty(len(v), dtype=np.complex128)
    out.real, out.imag = v["x"], v["y"]
    return out


def round_c64(z):
    """rounds c128 to match c64 precision and flushes subnormals to zero"""
    re, im = np.float64(f32(z.real)), np.float64(f32(z.imag))
    re[np.abs(re) < FLT_MIN] = 0.0
    im[np.abs(im) < FLT_MIN] = 0.0
    return re + 1j * im


def ulp_err(y, ref, floor=0.0):
    """
    Error of float32 results y in units of float32 ulp of the float64 reference.
    For references in the subnormal range the ulp is evaluated at FLT_MIN, i.e.
    flushing subnormals to zero (allowed in Vulkan) amounts to at most ~1 ulp.
    The ulp can be further limited from below by floor.
    References overflowing float32 accept inf or the largest finite value.
    """
    y, ref = f64(y), f64(ref)
    with np.errstate(over="ignore", invalid="ignore"):
        ref32 = ref.astype(np.float32)  # rounds to inf beyond FLT_MAX
        ref = np.where(np.isinf(ref32), ref32, ref)
        ulp = np.maximum(np.spacing(np.abs(ref32)), floor)
        err = np.abs(y - ref) / ulp
    # exact matches (incl. infinities)
    err[y == ref] = 0.0
    # overflow: Vulkan allows to return inf or the largest finite value
    overflow = np.isinf(ref) & (np.abs(y) >= FLT_MAX) & (np.sign(y) == np.sign(ref))
    err[overflow] = 0.0
    return err


def assert_complex(got, ref, max_ulp):
    """
    componentwise ulp error. Intermediate results may underflow to subnormals,
    which Vulkan allows to flush to zero, so close to FLT_MIN the error is
    measured in units of FLT_MIN instead.
    """
    got = c128(got)
    assert ulp_err(got.real, ref.real, floor=FLT_MIN).max() <= max_ulp
    assert ulp_err(got.imag, ref.imag, floor=FLT_MIN).max() <= max_ulp


def logUniform(rng, lo, hi, n, signed=True):
    """|x| log-uniformly distributed in [10^lo, 10^hi]"""
    x = 10.0 ** rng.uniform(lo, hi, n)
    if signed:
        x *= rng.choice([-1.0, 1.0], n)
    return x


def rndPolar(rng, lo, hi, n):
    """random complex numbers with log-uniform magnitude and uniform angle"""
    r = 10.0 ** rng.uniform(lo, hi, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    return r * np.exp(1j * phi)


def rndHuge(rng, n):
    """
    max(|x|, |y|) >= 1.5 * 2^126, where 1/max(|x|, |y|) is subnormal.
    |a| may exceed FLT_MAX while both components are finite.
    """
    z = logUniform(rng, np.log10(1.5 * 2.0**126), 38.53, n)
    z = z + 1j * logUniform(rng, 0, 38.53, n)
    z[::2] *= 1j
    return z


def matches(got, expected, tol=0.0):
    """
    Compares a single result with an entry of the tables of special values:
    nonzero finite values must match within tol ulp, infinities exactly and
    NaN any NaN. +0.0 accepts a zero of either sign, -0.0 only -0.0, as signed
    zeros are only guaranteed where they are produced by bit operations.
    """
    if expected is POVF or expected is NOVF:
        return abs(got) >= FLT_MAX and (got > 0) == (expected is POVF)
    if expected is TINY:
        return 0.0 <= got < FLT_MIN
    if np.isnan(expected):
        return bool(np.isnan(got))
    if np.isinf(expected) or expected == 0.0:
        return got == expected and (np.signbit(got) or not np.signbit(expected))
    return ulp_err(np.array([got]), np.array([expected]))[0] <= tol


def runShader(shader, TQuery, TResult, queries):
    N = len(queries)
    N_padded = -(-N // 32) * 32

    query_buffer = hp.ArrayBuffer(TQuery, N_padded)
    query_tensor = hp.ArrayTensor(TQuery, N_padded)
    result_buffer = hp.ArrayBuffer(TResult, N_padded)
    result_tensor = hp.ArrayTensor(TResult, N_padded)
    # we only expect floats as inputs
    flat = structured_to_unstructured(queries).astype(np.float32)
    q = query_buffer.numpy().view(np.float32).reshape(N_padded, -1)
    q[:N] = flat

    program = hp.Program(compileShader(shader))
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    hp.beginSequence().AndList(
        [
            hp.updateTensor(query_buffer, query_tensor),
            program.dispatch(N_padded // 32),
            hp.retrieveTensor(result_tensor, result_buffer),
        ]
    ).Submit().wait()

    # return results
    return result_buffer.numpy()[:N].copy()


def splitTestResults(cases, results):
    out, start = {}, 0
    for name, n in cases:
        out[name] = results[start : start + n]
        start += n
    return out


#################################### SCALAR ####################################


# For each binade of [1, 2^128) the float32 x closest to a multiple of pi/2
# among candidates derived from the continued fraction expansion of
# 2^(e-23) * 2/pi mod 1 (found with mpmath at 400 bits; includes the known
# worst case 16367173 * 2^72 = 0x6F79BE45). They stress the argument reduction,
# as the result is many orders of magnitude smaller than the argument.
# numpy's float64 sin/cos agree with mpmath on them to far below a float32 ulp.
# fmt: off
SINCOS_HARD = [
    0x3FC90FDB, 0x40490FDA, 0x4096CBE4, 0x4116CBE4, 0x4196CBE4, 0x4216CBE4, 0x4296CBE4,
    0x437CE5F1, 0x43FCE5F1, 0x447CE5F1, 0x44FCE5F1, 0x450BE628, 0x458BE628, 0x460BE628,
    0x468BE628, 0x474D246F, 0x47CD246F, 0x484D246F, 0x4882665E, 0x4902665E, 0x4982665E,
    0x4A2562AE, 0x4AA562AE, 0x4B2562AE, 0x4BF3B47B, 0x4C2332E9, 0x4CA332E9, 0x4D2332E9,
    0x4D847661, 0x4E13D4A5, 0x4E93D4A5, 0x4F0FFD14, 0x4F8FFD14, 0x500FFD14, 0x50A3E87F,
    0x5123E87F, 0x51A3E87F, 0x5223E87F, 0x52A3E87F, 0x5323E87F, 0x53B146A6, 0x543146A6,
    0x54B146A6, 0x553146A6, 0x55B146A6, 0x56787577, 0x569FD224, 0x571FD224, 0x57B82989,
    0x58103500, 0x58903500, 0x59103500, 0x598731B0, 0x5A02B008, 0x5A806F34, 0x5B006F34,
    0x5B806F34, 0x5C07BCD0, 0x5C87BCD0, 0x5D07BCD0, 0x5D87BCD0, 0x5E07BCD0, 0x5E87BCD0,
    0x5F07BCD0, 0x5FE4112C, 0x6064112C, 0x60AB0CE1, 0x612B0CE1, 0x61D3B126, 0x6253B126,
    0x62EC1B4A, 0x636C1B4A, 0x63E600C1, 0x642E0733, 0x64AE0733, 0x652E0733, 0x65898498,
    0x66098498, 0x66898498, 0x67098498, 0x67898498, 0x68098498, 0x68898498, 0x6946E3BB,
    0x69C6E3BB, 0x6A1976F1, 0x6A9976F1, 0x6B1976F1, 0x6B9976F1, 0x6C55DA58, 0x6CD5DA58,
    0x6D2063C2, 0x6D85A877, 0x6E05A877, 0x6E85A877, 0x6F79BE45, 0x6FF9BE45, 0x7079BE45,
    0x70F9BE45, 0x7179BE45, 0x71F9BE45, 0x723FA09A, 0x72BFA09A, 0x733FA09A, 0x73BFA09A,
    0x7452DE59, 0x74D2DE59, 0x756FA1DC, 0x75EFA1DC, 0x7633B965, 0x76A426EB, 0x77584625,
    0x77D84625, 0x78584625, 0x78A8B883, 0x79407F54, 0x79C07F54, 0x7A105F7F, 0x7A905F7F,
    0x7B1675C0, 0x7B9675C0, 0x7C1675C0, 0x7C83A708, 0x7D03A708, 0x7D83A708, 0x7E014D31,
    0x7EBDCDA0, 0x7F3DCDA0,
]
# fmt: on


def _scalarCases():
    N = 32 * 1024
    rng = np.random.default_rng(0xC0FFEE)
    hard = np.array(SINCOS_HARD, dtype=np.uint32).view(np.float32)
    cases = {
        # full range without overflow
        "expf": rng.uniform(-87.3, 88.72, N),
        # relative accuracy close to zero is the whole point of expm1
        "expm1": np.concatenate(
            [logUniform(rng, -30, 0, N), rng.uniform(-87.3, 88.72, N)]
        ),
        "log1p": np.concatenate(
            [
                logUniform(rng, -30, 0, N) * 0.999,
                logUniform(rng, 0, 38, N, signed=False),
                -1.0 + logUniform(rng, -7.2, 0, N // 4, signed=False),
            ]
        ),
        # covers polynomial (<1), expm1/expf (<logf(FLT_MAX)) and scaled branch
        "hyperbolic": np.concatenate(
            [
                logUniform(rng, -30, 0, N),
                rng.uniform(-89.41, 89.41, N),
                rng.uniform(0.9, 1.1, N // 8) * rng.choice([-1.0, 1.0], N // 8),
                rng.uniform(88.6, 89.41, N // 8) * rng.choice([-1.0, 1.0], N // 8),
            ]
        ),
        # fast path (Cody-Waite), switch over at 71476 and slow path (Payne-Hanek)
        "sincos": np.concatenate(
            [
                rng.uniform(-2.0 * np.pi, 2.0 * np.pi, N),
                logUniform(rng, -30, 38.5, N),
                rng.uniform(7.1e4, 7.2e4, N // 8) * rng.choice([-1.0, 1.0], N // 8),
            ]
        ),
        "sincos_hard": np.concatenate([hard, -hard]),
    }
    return {k: f32(v) for k, v in cases.items()}


class ScalarQuery(Structure):
    _fields_ = [("x", c_float)]


class ScalarResult(Structure):
    _fields_ = [
        ("expf", c_float),
        ("expm1", c_float),
        ("log1p", c_float),
        ("sinhf", c_float),
        ("coshf", c_float),
        ("sinhcoshf", vec2),
        ("sincos", vec2),
    ]


@pytest.fixture(scope="module")
def scalarResults():
    cases = _scalarCases()
    x = np.concatenate(list(cases.values()))
    queries = np.zeros(len(x), dtype=[("x", np.float32)])
    queries["x"] = x
    res = runShader("math.scalar.test.glsl", ScalarQuery, ScalarResult, queries)
    res = splitTestResults([(k, len(v)) for k, v in cases.items()], res)
    return {k: (cases[k], res[k]) for k in cases}


def test_signBit(rng):
    N = 32 * 256

    # allocate memory
    query_buffer = hp.FloatBuffer(N)
    query_tensor = hp.FloatTensor(N)
    result_buffer = hp.FloatBuffer(N)
    result_tensor = hp.FloatTensor(N)

    # fill queries
    floats = (rng.random(N, dtype=np.float32) - 0.5) * 1e36
    # special values: +/- 0.0
    floats[0] = 0.0
    floats[1] = -0.0
    query_buffer.numpy()[:] = floats

    # create test program
    program = hp.Program(compileShader("math.signBit.test.glsl"))
    program.bindParams(FloatIn=query_tensor, FloatOut=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = result_buffer.numpy()

    # check result
    assert np.all(results == np.copysign(1.0, floats))


@requiresFma
def test_expf(scalarResults):
    x, r = scalarResults["expf"]
    assert ulp_err(r["expf"], np.exp(f64(x))).max() <= 1.0


@requiresFma
def test_expm1(scalarResults):
    x, r = scalarResults["expm1"]
    assert ulp_err(r["expm1"], np.expm1(f64(x))).max() <= 1.0


@requiresFma
def test_log1p(scalarResults):
    x, r = scalarResults["log1p"]
    assert ulp_err(r["log1p"], np.log1p(f64(x))).max() <= 1.0


@requiresFma
def test_sinhf(scalarResults):
    x, r = scalarResults["hyperbolic"]
    ref = np.sinh(np.float64(x))
    assert ulp_err(r["sinhf"], ref).max() <= 2.0
    assert ulp_err(r["sinhcoshf"]["x"], ref).max() <= 2.0


@requiresFma
def test_coshf(scalarResults):
    x, r = scalarResults["hyperbolic"]
    ref = np.cosh(np.float64(x))
    assert ulp_err(r["coshf"], ref).max() <= 2.0
    assert ulp_err(r["sinhcoshf"]["y"], ref).max() <= 2.0


@requiresFma
def test_sincos(scalarResults):
    x, r = scalarResults["sincos"]
    f64 = np.float64(x)
    assert ulp_err(r["sincos"]["x"], np.sin(f64)).max() <= 1.5
    assert ulp_err(r["sincos"]["y"], np.cos(f64)).max() <= 1.5


@requiresFma
def test_sincos_hardCases(scalarResults):
    x, r = scalarResults["sincos_hard"]
    f64 = np.float64(x)
    assert ulp_err(r["sincos"]["x"], np.sin(f64)).max() <= 1.5
    assert ulp_err(r["sincos"]["y"], np.cos(f64)).max() <= 1.5


# fmt: off
SCALAR_COLUMNS = ["expf", "expm1", "log1p", "sinhf", "coshf", "sin", "cos"]
SCALAR_SPECIAL = [
    # x, expf, expm1, log1p, sinh, cosh, sin, cos  (None: not checked)
    # -0 is passed through by bit operations/selects in expm1, log1p, sinh, sin
    (-0.0,    1.0, -0.0, -0.0, -0.0, 1.0,  -0.0,  1.0),
    # expf: result in the subnormal range
    (-89.5,  TINY, None, None, None, None, None, None),
    # expf: |x| >= 104 branch; expm1: exp2 branch; sinh, cosh: sign of overflow
    (200.0,  POVF, POVF, None, POVF, POVF, None, None),
    (-200.0, 0.0,  -1.0, None, NOVF, POVF, None, None),
]
# fmt: on


@requiresFma
def test_scalar_specialValues():
    queries = np.empty(len(SCALAR_SPECIAL), dtype=[("x", np.float32)])
    queries["x"] = [row[0] for row in SCALAR_SPECIAL]
    r = runShader("math.scalar.test.glsl", ScalarQuery, ScalarResult, queries)
    results = {
        "expf": [("expf", r["expf"])],
        "expm1": [("expm1", r["expm1"])],
        "log1p": [("log1p", r["log1p"])],
        "sinhf": [("sinhf", r["sinhf"]), ("sinhcoshf.x", r["sinhcoshf"]["x"])],
        "coshf": [("coshf", r["coshf"]), ("sinhcoshf.y", r["sinhcoshf"]["y"])],
        "sin": [("sin", r["sincos"]["x"])],
        "cos": [("cos", r["sincos"]["y"])],
    }
    failures = []
    for j, col in enumerate(SCALAR_COLUMNS, start=1):
        for name, got in results[col]:
            for i, row in enumerate(SCALAR_SPECIAL):
                if row[j] is not None and not matches(float(got[i]), row[j]):
                    failures.append(f"{name}({row[0]}) = {got[i]}, expected {row[j]}")
    assert not failures, "\n".join(failures)


@requiresFma
def test_overflowThresholds():
    """Last finite / first overflowing result around the overflow threshold"""
    # largest float32 x with exp(x) < FLT_MAX is 0x42B17217 (88.72283)
    x = np.array([0x42B17217, 0x42B17218], dtype=np.uint32).view(np.float32)
    # largest float32 x with sinh(x), cosh(x) < FLT_MAX: 89.41598 (0x42B2D4FC)
    y = np.array([0x42B2D4FC, 0x42B2D4FD], dtype=np.uint32).view(np.float32)
    queries = np.zeros(4, dtype=[("x", np.float32)])
    queries["x"] = np.concatenate([x, y])
    r = runShader("math.scalar.test.glsl", ScalarQuery, ScalarResult, queries)
    # values right below the threshold must be finite and accurate
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    assert ulp_err(r["expf"][:1], np.exp(x64[:1])).max() <= 1.0
    assert ulp_err(r["expm1"][:1], np.expm1(x64[:1])).max() <= 1.0
    for name, ref in [("sinhf", np.sinh(y64)), ("coshf", np.cosh(y64))]:
        assert ulp_err(r[name][2:3], ref[:1]).max() <= 2.0
    assert ulp_err(r["sinhcoshf"]["x"][2:3], np.sinh(y64[:1])).max() <= 2.0
    assert ulp_err(r["sinhcoshf"]["y"][2:3], np.cosh(y64[:1])).max() <= 2.0
    # one ulp above must overflow (inf or FLT_MAX)
    assert r["expf"][1] >= FLT_MAX and r["expm1"][1] >= FLT_MAX
    assert r["sinhf"][3] >= FLT_MAX and r["coshf"][3] >= FLT_MAX
    assert np.all(np.array(r["sinhcoshf"][3].tolist()) >= FLT_MAX)


#################################### COMPLEX ###################################


# Error guarantees provided by Vulkan. Some math routines ues them and thus
# their error depend on those as well. See:
# https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
VK_DIV = 2.5  # x / y for |y| in [2^-126, 2^126]
VK_SQRT = 5.0  # inherited from 1.0 / inversesqrt() with inversesqrt 2 ulp
VK_LOG = 3.0  # outside [0.5, 2]; absolute error < 2^-21 inside
ATAN = 16.0  # conservative estimate of actual error
# TODO: Vulkan provides only a very loose error bound of 4096 ulps for atan.
#       In reality, most implementation have a much lower of only a few ulps.
#       It might be worthwhile to replace it with a custom implementation that
#       provides stronger guarantees.


# Expected error for complex functions (usually in ulps)
# Note that these rely on correctly fused fma instructions
CABS = 7.0  # sqrt(1 + n^2): VK_SQRT + n = min/max (<= VK_DIV / 2) + fma/mul
CSQRT = 12.0  # rho = sqrt(|x|/2 + |a|/2): (CABS + 1) / 2 + VK_SQRT; |y|/2rho: + VK_DIV
CPYTH = 13.0  # CSQRT + rounding of 1 - z^2
CMUL = 1.5  # prodDiff (error-free transformation)
CDIV = 6.0  # prodDiff 1.5 + |b|^2 1.5 + VK_DIV
CEXP = 3.0  # expf 1 + sincos 1.5 + product 0.5
CTRIG = 4.0  # sincos 1.5 + sinhcosh 1.5 + product 0.5
CTAN = 10.0  # numerator 3.5 + denominator 4 + VK_DIV
# log|a| = log(cabs(a)): VK_LOG + relative error of cabs, which becomes absolute
# (< 2 ulp for |log|a|| >= 2). Close to 1 the built-in log() is only accurate to
# 2^-21 absolute, plus again the error of cabs.
CLOG = 5.0
CLOG_ATOL = 2.0**-20
CLOG1P = 2.0
# cpow = cexp(b * clog(a)): the errors of clog (CLOG + ATAN ulp, CLOG_ATOL = 8 EPS)
# get multiplied by |b|, plus the product 2 and CEXP
CPOW = 26.0


def _complexCases(rng):
    N = 16 * 1024
    cases = {}
    # |a|^2 over-/underflows for most of the range
    cases["cabs"] = (
        np.concatenate(
            [
                rndPolar(rng, -37.5, 38.53, N),
                rndHuge(rng, N // 4),
                # very different magnitudes of the components
                logUniform(rng, 10, 30, N) + 1j * logUniform(rng, -30, -10, N),
            ]
        ),
        None,
    )
    # cmul: random and with heavy cancellation in real / imag part
    a = rndPolar(rng, -10, 10, 3 * N)
    a, b = round_c64(a), round_c64(rndPolar(rng, -10, 10, 3 * N))
    eps = logUniform(rng, -7, -2, N)
    b[N : 2 * N] = round_c64((a[N : 2 * N].imag + 1j * a[N : 2 * N].real) * (1 + eps))
    b[2 * N :] = round_c64((a[2 * N :].imag - 1j * a[2 * N :].real) * (1 + eps))
    cases["cmul"] = (a, b)
    # cdiv: wide exponent range; scaling must avoid over/underflow of |b|^2.
    # The last two segments produce overflowing / subnormal results.
    a = np.concatenate(
        [
            rndPolar(rng, -15, 15, N),
            rndPolar(rng, 36, 38, N // 4),
            rndPolar(rng, -37, -35, N // 4),
            rndPolar(rng, 0, 38, N // 4),
            rndPolar(rng, -37, 0, N // 4),
        ]
    )
    b = np.concatenate(
        [
            rndPolar(rng, -15, 15, N),
            rndPolar(rng, 36, 38, N // 4),
            rndPolar(rng, -37, -35, N // 4),
            rndPolar(rng, -37, 0, N // 4),
            rndPolar(rng, 0, 38, N // 4),
        ]
    )
    cases["cdiv"] = (a, b)
    cases["csqrt"] = (
        np.concatenate(
            [
                rndPolar(rng, -37.5, 38.53, N),
                rndHuge(rng, N // 4),
                # close to the branch cut
                -logUniform(rng, -10, 10, N // 4, signed=False)
                + 1j * logUniform(rng, -30, -10, N // 4),
            ]
        ),
        None,
    )
    cases["cpyth"] = (
        np.concatenate(
            [
                rndPolar(rng, -3, 1, N),
                # close to +/-1 where 1 - z^2 suffers from cancellation
                (1 + logUniform(rng, -7, -1, N)) * rng.choice([-1.0, 1.0], N)
                + 1j * logUniform(rng, -10, -1, N),
            ]
        ),
        None,
    )
    cases["cexp"] = (
        np.concatenate(
            [
                rng.uniform(-80, 80, N) + 1j * rng.uniform(-100, 100, N),
                rng.uniform(-5, 5, N // 4) + 1j * logUniform(rng, 3, 7, N // 4),
            ]
        ),
        None,
    )
    cases["clog"] = (
        np.concatenate(
            [
                rndPolar(rng, -37.5, 38.53, N),
                rndHuge(rng, N // 4),
                # |a| in [0.9, 1.1], where clog uses log1p
                (1 + logUniform(rng, -7, -1, N))
                * np.exp(1j * rng.uniform(-np.pi, np.pi, N)),
            ]
        ),
        None,
    )
    a = rndPolar(rng, -3, 3, N)
    b = rng.uniform(-6, 6, N) + 1j * rng.uniform(-6, 6, N)
    cases["cpow"] = (a, b)
    trig = np.concatenate(
        [
            rng.uniform(-10, 10, N) + 1j * logUniform(rng, -10, 1, N),
            rng.uniform(-10, 10, N) + 1j * rng.uniform(-89, 89, N),
            logUniform(rng, 1, 4, N // 4) + 1j * rng.uniform(-5, 5, N // 4),
        ]
    )
    cases["ctrig"] = (trig, None)
    cases["ctan"] = (
        np.concatenate(
            [
                rng.uniform(-10, 10, N) + 1j * logUniform(rng, -10, 2.5, N),
                # around the switch at |y| = 10
                rng.uniform(-10, 10, N // 4) + 1j * rng.uniform(-11, 11, N // 4),
            ]
        ),
        None,
    )
    out = {}
    for k, (a, b) in cases.items():
        a = round_c64(a)
        b = round_c64(b) if b is not None else np.zeros_like(a)
        out[k] = (a, b)
    return out


class ComplexQuery(Structure):
    _fields_ = [("a", vec2), ("b", vec2)]


class ComplexResult(Structure):
    _fields_ = [
        ("cabs", c_float),
        ("carg", c_float),
        ("cmul", vec2),
        ("cdiv", vec2),
        ("csqrt", vec2),
        ("cpyth", vec2),
        ("cexp", vec2),
        ("clog", vec2),
        ("cpowf", vec2),
        ("cpow", vec2),
        ("csin", vec2),
        ("ccos", vec2),
        ("ctan", vec2),
    ]


@pytest.fixture(scope="module")
def complexResults():
    rng = np.random.default_rng(0xC0FFEE)
    cases = _complexCases(rng)
    a = np.concatenate([v[0] for v in cases.values()])
    b = np.concatenate([v[1] for v in cases.values()])
    queries = np.zeros(len(a), dtype=[("a", np.float32, 2), ("b", np.float32, 2)])
    queries["a"] = np.stack([a.real, a.imag], -1)
    queries["b"] = np.stack([b.real, b.imag], -1)
    res = runShader("math.complex.test.glsl", ComplexQuery, ComplexResult, queries)
    res = splitTestResults([(k, len(v[0])) for k, v in cases.items()], res)
    return {k: (*cases[k], res[k]) for k in cases}


@requiresFma
def test_cabs(complexResults):
    a, _, r = complexResults["cabs"]
    assert ulp_err(r["cabs"], np.abs(a)).max() <= CABS


def test_carg(complexResults):
    # no fused multiply-add needed; also covers the imaginary part of clog
    a, _, r = complexResults["clog"]
    # angles below FLT_MIN may be flushed to zero
    assert ulp_err(r["carg"], np.angle(a), floor=FLT_MIN).max() <= ATAN


@requiresFma
def test_cmul(complexResults):
    a, b, r = complexResults["cmul"]
    assert_complex(r["cmul"], a * b, CMUL)


@requiresFma
def test_cdiv(complexResults):
    a, b, r = complexResults["cdiv"]
    assert_complex(r["cdiv"], a / b, CDIV)


@requiresFma
def test_csqrt(complexResults):
    a, _, r = complexResults["csqrt"]
    assert_complex(r["csqrt"], np.sqrt(a), CSQRT)


@requiresFma
def test_cpyth(complexResults):
    a, _, r = complexResults["cpyth"]
    assert_complex(r["cpyth"], np.sqrt(1.0 - a * a), CPYTH)


@requiresFma
def test_cexp(complexResults):
    a, _, r = complexResults["cexp"]
    assert_complex(r["cexp"], np.exp(a), CEXP)


@requiresFma
def test_clog(complexResults):
    a, _, r = complexResults["clog"]
    got, ref = c128(r["clog"]).real, np.log(np.abs(a))  # imaginary part: test_carg
    assert ulp_err(got, ref, floor=CLOG_ATOL / CLOG).max() <= CLOG
    # close to the unit circle log|a| is small and must stay relatively accurate
    near = np.abs(np.abs(a) - 1.0) <= 0.1
    assert ulp_err(got[near], ref[near]).max() <= CLOG1P


@requiresFma
def test_cpow(complexResults):
    a, b, r = complexResults["cpow"]
    for name, e in [("cpowf", b.real), ("cpow", b)]:
        tol = CPOW * EPS * (1.0 + np.abs(e) + np.abs(e * np.log(a)))
        got, ref = c128(r[name]), a**e
        assert np.all(np.abs(got - ref) / np.abs(ref) <= tol)


@requiresFma
def test_csin_ccos(complexResults):
    a, _, r = complexResults["ctrig"]
    assert_complex(r["csin"], np.sin(a), CTRIG)
    assert_complex(r["ccos"], np.cos(a), CTRIG)


@requiresFma
def test_ctan(complexResults):
    a, _, r = complexResults["ctan"]
    assert_complex(r["ctan"], np.tan(a), CTAN)


# fmt: off
PI = float(f32(np.pi))
COMPLEX_SPECIAL = [
    # func,        a,          b,         expected,  err (ulp)
    # carg: imaginary part zero branch, all combinations of signed zeros
    ("carg",  ( 0.0,  0.0),       None,         0.0,   0.0),
    ("carg",  ( 0.0, -0.0),       None,        -0.0,   0.0),
    ("carg",  (-0.0,  0.0),       None,          PI,   0.0),
    ("carg",  (-0.0, -0.0),       None,         -PI,   0.0),
    # zero branches
    ("cabs",   (0.0, -0.0),       None,         0.0,   0.0),
    ("csqrt",  (0.0, -0.0),       None, (0.0, -0.0),   0.0),
    ("cpow",   (0.0,  0.0), (2.0, 1.0), (0.0,  0.0),   0.0),
    ("cpow",   (0.0,  0.0), (0.0, 0.0), (1.0,  0.0),   0.0),
    ("cpowf",  (0.0,  0.0), (2.0, 0.0), (0.0,  0.0),   0.0),
    ("cpowf",  (0.0,  0.0), (0.0, 0.0), (1.0,  0.0),   0.0),
    # sign of zero on the branch cuts
    ("csqrt", ( 4.0, -0.0),       None, (2.0, -0.0), CSQRT),
    ("csqrt", (-4.0,  0.0),       None, (0.0,  2.0), CSQRT),
    ("csqrt", (-4.0, -0.0),       None, (0.0, -2.0), CSQRT),
    ("clog",  (-1.0,  0.0),       None, (0.0,   PI),   0.0),
    ("clog",  (-1.0, -0.0),       None, (0.0,  -PI),   0.0),
    # results exactly zero (infinite relative error otherwise)
    ("cpyth", ( 1.0,  0.0),       None, (0.0,  0.0),   0.0),
    ("clog",  ( 1.0,  0.0),       None, (0.0,  0.0),   0.0),
]
# fmt: on


@requiresFma
def test_complex_specialValues():
    N = len(COMPLEX_SPECIAL)
    queries = np.zeros(N, dtype=[("a", np.float32, 2), ("b", np.float32, 2)])
    for i, (_, a, b, *_) in enumerate(COMPLEX_SPECIAL):
        queries["a"][i] = a
        queries["b"][i] = b if b is not None else (0.0, 0.0)
    r = runShader("math.complex.test.glsl", ComplexQuery, ComplexResult, queries)
    failures = []
    for i, (fn, a, b, expected, tol) in enumerate(COMPLEX_SPECIAL):
        got = np.atleast_1d(r[fn][i].tolist())
        if not all(matches(g, e, tol) for g, e in zip(got, np.atleast_1d(expected))):
            args = f"{a}" if b is None else f"{a}, {b}"
            failures.append(f"{fn}({args}) = {tuple(got)}, expected {expected}")
    assert not failures, "\n".join(failures)

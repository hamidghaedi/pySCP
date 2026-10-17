"""The significance layer's tests and labels, pinned against R.

The fixture is written by ``tools/parity/feature_stat_tests.R`` from base R's
``wilcox.test`` / ``t.test`` / ``pairwise.*.test`` / ``kruskal.test`` /
``anova(lm())`` and from ``format.pval`` / ``symnum`` / ``sprintf`` -- the calls
ggpubr and ggsignif make.  The suite does not need R.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scp.pl import _stat_sig as sig

FIX = Path(__file__).parent / "fixtures" / "parity" / "feature_stat" / "stat_tests.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(FIX.read_text())


def _arr(v):
    return np.atleast_1d(np.asarray(v, dtype=float))


def _close(py, r):
    if r is None:
        return math.isnan(py)
    return py == pytest.approx(r, rel=1e-9, abs=1e-15)


def test_wilcox_matches_r_default_exact_and_normal_branches(golden):
    for name, case in golden["wilcox"].items():
        x, y = _arr(case["x"]), _arr(case["y"])
        assert _close(sig.wilcox_test(x, y), case["p"]), name
        assert _close(sig.wilcox_test(x, y, exact=False), case["p_exact_false"]), name
        # pairwise.wilcox.test is wilcox.test pair by pair
        got = sig.pairwise_p([x, y], "wilcox.test")
        assert _close(got[0][2], case["p_pairwise"]), name


def test_exact_branch_is_taken_only_without_ties_and_under_50(golden):
    w = golden["wilcox"]
    # no ties, both < 50: exact differs from the normal approximation
    assert w["exact_small"]["p"] != pytest.approx(w["exact_small"]["p_exact_false"])
    assert w["exact_49_10"]["p"] != pytest.approx(w["exact_49_10"]["p_exact_false"])
    # ties (small) and n >= 50 both fall back to the approximation
    for name in ("ties_small", "normal_50_55", "normal_ties_large"):
        assert w[name]["p"] == pytest.approx(w[name]["p_exact_false"])


def test_welch_t_matches_r(golden):
    for name, case in golden["ttest"].items():
        x, y = _arr(case["x"]), _arr(case["y"])
        assert _close(sig.t_test(x, y), case["p"]), name
        assert _close(sig.pairwise_p([x, y], "t.test")[0][2], case["p_pairwise"]), name


def test_kruskal_and_anova_match_r(golden):
    for name, case in golden["multigroup"].items():
        groups = [_arr(g) for g in case["groups"]]
        assert _close(sig.kruskal_test(groups), case["kruskal"]), name
        assert _close(sig.anova_test(groups), case["anova"]), name


def test_p_value_text_matches_r(golden):
    for case in golden["formatting"]:
        p = list(_arr(case["p"]))
        want_fmt = case["format_pval"]
        want_fmt = [want_fmt] if isinstance(want_fmt, str) else want_fmt
        assert sig.format_pval(p) == want_fmt, p
        want_sym = case["symnum"]
        want_sym = [want_sym] if isinstance(want_sym, str) else want_sym
        assert sig.symnum_signif(p) == want_sym, p
        want_2g = case["sprintf_2g"]
        want_2g = [want_2g] if isinstance(want_2g, str) else want_2g
        assert [f"{v:.2g}" for v in p] == want_2g, p


def test_the_two_star_maps_disagree_at_the_boundary():
    """compare_means closes intervals on the right; ggsignif uses a strict cut."""
    assert sig.symnum_signif([0.05]) == ["*"]
    assert sig.signif_strict(0.05) == "ns"
    assert sig.signif_strict(0.0499) == "*"
    assert sig.signif_strict(float("nan")) == "NS."
    assert sig.symnum_signif([float("nan")]) == ["?"]


def test_r_errors_become_nan():
    assert math.isnan(sig.t_test([1.0], [1.0, 2.0]))
    assert math.isnan(sig.kruskal_test([[1.0, 2.0]]))
    assert math.isnan(sig.wilcox_test([], [1.0]))


def test_method_aliases_follow_ggpubr():
    assert sig.resolve_method("wilcox") == "wilcox.test"
    assert sig.resolve_method("kruskal") == "kruskal.test"
    assert sig.resolve_method("aov") == "anova"
    with pytest.raises(ValueError, match="unsupported"):
        sig.resolve_method("mann-whitney")

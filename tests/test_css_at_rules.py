"""Unit tests for the CSS at-rule stripper behind the selector-liveness audit.

Why these are here and not in ``tests/e2e/test_browser_selectors.py``: that module is
``-m browser`` behind an ``importorskip``, so a hand-rolled parser would be exercised only inside
the three-browser job. Parsing needs no page, and a silent parser regression shrinks what the audit
covers without turning anything red — so it is checked in the lane that runs on every push.

Every case below is a real failure mode, not a hypothetical: the two marked REGRESSION were
measured against the first version of this parser by review, and both were silent.
"""

from __future__ import annotations

import pytest

from app_ui.theme import CUSTOM_CSS
from tests.css_at_rules import close_brace, parse_selectors, scan, strip_at_rules

# The audit's coverage is exactly this set. Pinned because the only previous check on parser output
# was `assert selectors` — non-emptiness, which 1 of 14 satisfies. A parser bug that dropped twelve
# rules would have left the whole module green while auditing almost nothing.
EXPECTED_SELECTORS = [
    ".gradio-container",
    "footer",
    "#md-header",
    'label > span[data-testid="block-info"]',
    "button.primary",
    "#md-file",
    "#md-ppm .wrap",
    "#md-ppm label",
    "#md-ppm label.selected",
    "#md-examples button",
    "#md-examples button:hover",
    "#md-table table",
    "#md-table thead th",
    ".md-footnote",
]


def test_the_audited_selector_set_is_exactly_what_is_pinned() -> None:
    """Add a rule to CUSTOM_CSS and this list must grow with it — deliberately, not silently."""
    assert parse_selectors(CUSTOM_CSS) == EXPECTED_SELECTORS


def test_the_embedded_font_face_contributes_no_selectors() -> None:
    """`@font-face` declarations are not selectors.

    Leaving them in is what the browser rejected with
    `malformed selectors: ["font-family: 'Space Grotesk';…"]`.
    """
    parsed = parse_selectors(CUSTOM_CSS)
    assert not [s for s in parsed if ":" in s and not s.split(":", 1)[1][:1].isalpha()]
    assert not [s for s in parsed if "base64" in s or "font-family" in s or ";" in s]


@pytest.mark.parametrize(
    ("label", "css", "expected"),
    [
        (
            "REGRESSION: an `@` inside a string is data, not an at-rule",
            '.keepme { color: red; }\n.mailto::after { content: "a@b" }\n.also { color: blue; }',
            [".keepme", ".mailto::after", ".also"],
        ),
        (
            "REGRESSION: an unterminated statement at-rule must not eat the next rule",
            '@charset "utf-8"\n.keepme { color: red; }',
            [".keepme"],
        ),
        (
            "a declaration at-rule block goes whole, payload semicolons and all",
            "@font-face{font-family:X;src:url(data:font/woff2;base64,AAA)}\n.a{color:red}",
            [".a"],
        ),
        (
            "a conditional group rule is unwrapped, so its selectors stay audited",
            "@media (min-width:1px){.b{color:red}}",
            [".b"],
        ),
        (
            "group rules nest",
            "@supports (a:b){@media screen{.d{color:red}}}",
            [".d"],
        ),
        (
            "a `;` inside an @import URL does not truncate it",
            "@import url('http://x/y?w=1;2;3');\n.c{color:red}",
            [".c"],
        ),
        (
            "an unterminated block at EOF drops itself, not what came before",
            ".e{color:red}\n@font-face{font-family:X",
            [".e"],
        ),
        (
            "an unquoted url() containing `@` is not an at-rule",
            ".f{background:url(http://x/a@2x.png)}",
            [".f"],
        ),
        (
            "comments are stripped without eating rules",
            "/* @media (x) { .gone {a:b} } */\n.g{color:red}",
            [".g"],
        ),
    ],
)
def test_parse_selectors_handles(label: str, css: str, expected: list[str]) -> None:
    assert parse_selectors(css) == expected, label


def test_scan_ignores_delimiters_inside_strings() -> None:
    assert scan('a "x;y" ;', 0, ";") == 8  # the quoted `;` is skipped
    assert scan("no delimiter here", 0, ";") == -1


def test_close_brace_matches_nesting_and_survives_truncation() -> None:
    css = "@media(x){ .a{b:c} .d{e:f} }"
    assert close_brace(css, css.index("{")) == len(css) - 1
    assert close_brace("@media(x){ .a{b:c}", 9) == len("@media(x){ .a{b:c}")  # unterminated


def test_strip_at_rules_terminates_on_pathological_input() -> None:
    """Guards the recursion: group rules recurse on their body, which must always shrink."""
    for css in ("@media{" * 50, "@supports{@media{@layer{", "@media{}" * 100, "@", "@@@"):
        strip_at_rules(css)  # must return, not hang or recurse away

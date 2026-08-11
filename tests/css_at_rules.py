"""Pure-CSS parsing helpers for the selector-liveness audit.

Extracted out of ``tests/e2e/test_browser_selectors.py`` so a hand-rolled parser is not reachable
only through the three-browser job. The audit that consumes these needs a real page; the parsing
does not, and ``tests/test_css_at_rules.py`` exercises it in the lane that runs on every push.

Nothing here imports playwright or gradio.
"""

from __future__ import annotations

import re

# Conditional group rules: the body holds real rules, so unwrap rather than drop, or the selectors
# inside would stop being audited exactly when someone starts using them.
GROUP_AT_RULES = ("media", "supports", "layer", "container", "scope", "document")
# Bodies of declarations or keyframe stops, never page selectors — the whole block goes.
# `@font-face` is the one actually present (the embedded Space Grotesk face).
BLOCK_AT_RULES = (
    "font-face",
    "keyframes",
    "page",
    "property",
    "counter-style",
    "font-feature-values",
)
# Terminated by `;`, not by a block.
STATEMENT_AT_RULES = ("import", "charset", "namespace")


def scan(css: str, start: int, stops: str) -> int:
    """First index at/after `start` holding a char from `stops`, ignoring CSS strings; -1 if none.

    String-aware because `content: "a@b"` and `url('…;…')` both put a would-be delimiter inside a
    quoted value, and a naive `str.find` treats it as structure.
    """
    quote = ""
    i = start
    while i < len(css):
        char = css[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in stops:
            return i
        i += 1
    return -1


def close_brace(css: str, open_at: int) -> int:
    """Index of the `}` matching the `{` at `open_at`; `len(css)` if the block is unterminated."""
    depth, i, quote = 0, open_at, ""
    while i < len(css):
        char = css[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(css)


def strip_at_rules(css: str) -> str:
    """Drop `@…;` statements and declaration blocks; unwrap conditional group blocks.

    Two rules keep this from eating live selectors, both found by review rather than by the suite:

    * An `@` is only structure **outside** a string. An earlier version keyed on any `@`, so
      `content: "a@b"` with no trailing `;` made it brace-match to the *next* rule's block and
      delete that rule whole — silently, because the result was still valid CSS.
    * An unrecognised at-rule is left alone rather than guessed at. `@charset "utf-8"` missing its
      `;` otherwise consumed the following rule the same way. Leaving it produces a malformed
      selector, which `test_no_selector_is_syntactically_invalid` reports loudly.
    """
    out: list[str] = []
    i, quote = 0, ""
    while i < len(css):
        char = css[i]
        if quote:  # inside a string: `@` here is data, not structure
            if char == "\\" and i + 1 < len(css):
                out.append(css[i : i + 2])
                i += 2
                continue
            if char == quote:
                quote = ""
            out.append(char)
        elif char in "\"'":
            quote = char
            out.append(char)
        elif char != "@":
            out.append(char)
        else:
            match = re.match(r"@([a-zA-Z-]+)", css[i:])
            name = match.group(1).lower() if match else ""
            if name in STATEMENT_AT_RULES:
                # `;` *or* newline: an unterminated statement must not swallow the next rule.
                end = scan(css, i, ";\n")
                i = len(css) if end == -1 else end + 1
                continue
            if name in GROUP_AT_RULES or name in BLOCK_AT_RULES:
                open_at = scan(css, i, "{")
                if open_at == -1:
                    break  # unterminated at-rule at EOF; nothing after it to keep
                close_at = close_brace(css, open_at)
                if name in GROUP_AT_RULES:
                    out.append(strip_at_rules(css[open_at + 1 : close_at]))
                i = min(close_at + 1, len(css))
                continue
            out.append(char)  # unknown at-rule: fail loudly downstream, do not guess
        i += 1
    return "".join(out)


def parse_selectors(css: str) -> list[str]:
    """Selectors from a stylesheet, minus at-rules, comments, and pseudo-classes.

    Pseudo-classes are stripped by :func:`strip_pseudo` at the point of use rather than here, so
    `#md-examples button:hover` still verifies that `#md-examples button` exists — the half of the
    rule that can rot silently.
    """
    # Safe before at-rule handling only because the base64 font payload cannot contain `*`: the
    # base64 alphabet is A-Za-z0-9+/=, so no `/*` can appear inside it to open a phantom comment.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    css = strip_at_rules(css)

    selectors: list[str] = []
    for block in css.split("}"):
        head = block.split("{")[0].strip()
        if not head:
            continue
        selectors.extend(part.strip() for part in head.split(",") if part.strip())
    return selectors


def strip_pseudo(selector: str) -> str:
    """`#md-examples button:hover` -> `#md-examples button`; `a::before` -> `a`."""
    return re.sub(r"::?[a-zA-Z-]+(\([^)]*\))?", "", selector).strip()

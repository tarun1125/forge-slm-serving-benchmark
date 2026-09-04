"""Defense-in-depth guard around executing model-generated PyMongo queries.

FORGE executes LLM-generated strings against a LIVE Atlas cluster whose
credentials can write (forge.phase1.parity_check, forge.phase2.score_accuracy).
The primary control for that is the capstone harness's own
`check_query_is_safe()` + `safe_eval_query()` — an AST allowlist gating a raw
`eval()`. FORGE deliberately does not re-derive that harness (see
capstone_bridge.py), and this module does NOT do so either: it re-implements
none of the scoring, comparison, or evaluation logic. It is purely a second
gate in front of it.

Why a second gate exists at all — two verified gaps in the upstream allowlist,
confirmed empirically against the real function, not assumed:

1. The upstream pipeline-stage allowlist only inspects an `aggregate()` call
   whose first positional argument is a literal `ast.List`. Any other form
   silently skips stage validation entirely, so
   `db.c.aggregate(list([{"$out": "x"}]))` and
   `db.c.aggregate([{"$out": "x"}][0:1])` both pass the upstream check while
   still evaluating to a real list that pymongo will happily execute — i.e. a
   destructive write to the cluster. (A tuple form is also accepted upstream
   but happens to die later inside pymongo's own `validate_list`; relying on
   that coincidence is not a control.)

2. `$where` and `$function` are query operators, not pipeline stages, so they
   never reach the upstream stage check at all — `db.c.find({"$where": "..."})`
   passes it completely. Both execute server-side JavaScript.

Neither gap is reachable by the actual fine-tuned model in practice: all 2,526
real generations this project produced contain zero occurrences of any operator
denied here, so enabling this guard changes no published result. It exists so
that the safety of running generated code against a live database does not
depend on the model continuing to behave well.

Fails closed: anything it cannot prove safe is rejected.
"""

from __future__ import annotations

import ast

# Operators that write data or execute server-side JavaScript. Denied wherever
# they appear in the expression — as a pipeline stage, inside a find() filter,
# or nested at any depth — not only in the one shape the upstream check walks.
#
# Deliberately NOT denied: $expr, $size, $toInt/$toDouble and friends. Those are
# read-only, and the model's own training prompt explicitly instructs it to use
# $expr/$size for "at least N items" questions — denying them would reject
# legitimate, correct output.
DENIED_OPERATORS = frozenset(
    {
        "$out",  # writes/replaces a collection
        "$merge",  # writes into a collection
        "$where",  # server-side JavaScript
        "$function",  # server-side JavaScript
        "$accumulator",  # server-side JavaScript
        "$listSessions",
        "$listLocalSessions",
        "$currentOp",
        "$planCacheStats",
        "$collStats",
        "$indexStats",
    }
)


class UnsafeQueryError(ValueError):
    """Raised when a generated query is rejected before it can be executed."""


def _iter_string_constants(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def check_generated_query(query: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Does not execute anything.

    Reason is empty when safe. This runs BEFORE the capstone harness's own
    check, and is intentionally stricter — it is allowed to reject things the
    upstream check would permit, never the reverse.
    """
    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    # 1. Denied operators anywhere in the expression, at any nesting depth.
    #    Checked against every string constant rather than only dict keys: an
    #    operator smuggled in as a value (e.g. inside a $expr body) is just as
    #    executable, and no legitimate generated query needs these as data.
    for value in _iter_string_constants(tree):
        if value in DENIED_OPERATORS:
            return False, f"denied operator: {value}"

    # 2. An aggregate() pipeline must be a literal list, so that the upstream
    #    stage allowlist actually inspects it instead of silently skipping.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "aggregate"
            and node.args
            and not isinstance(node.args[0], ast.List)
        ):
            return False, (
                "aggregate() pipeline must be a literal list — a computed pipeline "
                "bypasses the upstream per-stage safety allowlist"
            )

    return True, ""


def assert_generated_query_safe(query: str) -> None:
    """check_generated_query(), raising UnsafeQueryError instead of returning.

    Call sites treat this the same way they already treat an upstream
    rejection: a rejected query is a real, scoreable wrong answer, not a crash.
    """
    ok, reason = check_generated_query(query)
    if not ok:
        raise UnsafeQueryError(f"REJECTED by forge.query_guard ({reason})")

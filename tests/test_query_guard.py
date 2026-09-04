"""Every rejection case below was first confirmed to PASS the upstream
capstone allowlist (evaluation/execute_queries.check_query_is_safe) — these
aren't hypothetical payloads, they're the real gaps forge.query_guard exists
to close. The legitimate-query cases are real shapes taken from this
project's own generated output and system prompt, to prove the guard doesn't
reject correct model behavior.
"""

import pytest

from forge.query_guard import (
    UnsafeQueryError,
    assert_generated_query_safe,
    check_generated_query,
)


class TestLegitimateQueriesStillPass:
    """The guard must not reject anything the model legitimately produces."""

    @pytest.mark.parametrize(
        "query",
        [
            'db.employees.find({}, {"_id": 0, "FIRST_NAME": 1})',
            'db.employees.aggregate([{"$group": {"_id": "$DEPARTMENT_ID", "n": {"$sum": 1}}}])',
            # The system prompt explicitly instructs list(...) wrapping.
            'list(db.singer.find({"age": {"$gt": 20}}, {"_id": 0}))',
            'db.employees.count_documents({"SALARY": {"$gt": 10000}})',
            # $expr/$size are explicitly taught by the training prompt for
            # "at least N items" questions — must stay allowed.
            'db.c.find({"$expr": {"$gte": [{"$size": "$items"}, 4]}}, {"_id": 0})',
            # $toDouble/$ne guard pattern the training prompt mandates.
            'db.cars_data.aggregate([{"$match": {"MPG": {"$ne": "null"}}}, '
            '{"$group": {"_id": None, "avg": {"$avg": {"$toDouble": "$MPG"}}}}])',
            'db["model_list.json"].find({}, {"_id": 0})',
        ],
    )
    def test_allows_real_generated_shapes(self, query):
        ok, reason = check_generated_query(query)
        assert ok, f"guard wrongly rejected a legitimate query: {reason}"


class TestDestructivePipelineBypasses:
    """Gap 1: upstream only inspects aggregate() pipelines that are literal
    lists, so a computed pipeline skips stage validation entirely."""

    @pytest.mark.parametrize(
        "query",
        [
            'db.employees.aggregate(list([{"$out": "pwned"}]))',
            'db.employees.aggregate([{"$out": "pwned"}][0:1])',
            'db.employees.aggregate(({"$merge": {"into": "pwned"}},))',
        ],
    )
    def test_rejects_computed_pipeline(self, query):
        ok, reason = check_generated_query(query)
        assert not ok
        assert "$out" in reason or "$merge" in reason or "literal list" in reason

    def test_rejects_out_even_in_literal_list(self, query='db.c.aggregate([{"$out": "x"}])'):
        ok, _ = check_generated_query(query)
        assert not ok


class TestServerSideJavaScript:
    """Gap 2: $where/$function are query operators, not pipeline stages, so
    upstream's stage allowlist never sees them at all."""

    @pytest.mark.parametrize(
        "query",
        [
            'db.employees.find({"$where": "while(true){}"})',
            'db.employees.count_documents({"$where": "1==1"})',
            'db.c.find({"$expr": {"$function": '
            '{"body": "function(){}", "args": [], "lang": "js"}}})',
            'db.c.aggregate([{"$group": {"_id": None, "v": {"$accumulator": {"lang": "js"}}}}])',
        ],
    )
    def test_rejects_server_side_javascript(self, query):
        ok, reason = check_generated_query(query)
        assert not ok, f"server-side JS slipped through: {query}"
        assert "denied operator" in reason


class TestGuardMechanics:
    def test_syntax_error_is_rejected_not_raised(self):
        ok, reason = check_generated_query("db.c.find({")
        assert not ok
        assert "syntax error" in reason

    def test_assert_raises_unsafe_query_error(self):
        with pytest.raises(UnsafeQueryError, match="denied operator"):
            assert_generated_query_safe('db.c.find({"$where": "1==1"})')

    def test_assert_is_silent_for_safe_query(self):
        assert_generated_query_safe('db.c.find({}, {"_id": 0})')

    def test_denied_operator_nested_deeply_is_still_caught(self):
        query = 'db.c.aggregate([{"$facet": {"a": [{"$out": "pwned"}]}}])'
        ok, reason = check_generated_query(query)
        assert not ok
        assert "$out" in reason

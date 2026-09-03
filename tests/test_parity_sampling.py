from forge.phase1.parity_check import stratified_sample


def _make_cases(complexity: str, n: int) -> list[dict]:
    return [{"id": f"{complexity}-{i}", "complexity": complexity} for i in range(n)]


def test_stratified_sample_returns_exactly_n():
    cases = _make_cases("easy", 100) + _make_cases("medium", 150) + _make_cases("hard", 54)
    sample = stratified_sample(cases, n=50, seed=42)
    assert len(sample) == 50


def test_stratified_sample_is_deterministic_for_a_given_seed():
    cases = _make_cases("easy", 100) + _make_cases("medium", 150) + _make_cases("hard", 54)
    sample_a = stratified_sample(cases, n=50, seed=42)
    sample_b = stratified_sample(cases, n=50, seed=42)
    assert [c["id"] for c in sample_a] == [c["id"] for c in sample_b]


def test_stratified_sample_covers_every_stratum_present():
    cases = _make_cases("easy", 100) + _make_cases("medium", 150) + _make_cases("hard", 54)
    sample = stratified_sample(cases, n=50, seed=42)
    complexities = {c["complexity"] for c in sample}
    assert complexities == {"easy", "medium", "hard"}


def test_stratified_sample_never_exceeds_available_cases_in_a_stratum():
    cases = _make_cases("easy", 5) + _make_cases("medium", 200)
    sample = stratified_sample(cases, n=50, seed=42)
    easy_count = sum(1 for c in sample if c["complexity"] == "easy")
    assert easy_count <= 5

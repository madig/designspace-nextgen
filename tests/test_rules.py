from designspace_nextgen import ConditionSet, Rule


def test_always_substitute() -> None:
    rule = Rule(
        name="always_apply",
        condition_sets=[ConditionSet(conditions=[])],
        substitutions={"a": "a.alt"},
    )
    assert rule.evaluate({"Weight": 400}, {"a", "a.alt"}) == [("a", "a.alt")]

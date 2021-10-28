from fontTools.misc import etree as ElementTree  # type: ignore

from designspace_nextgen import ConditionSet, Range, Rule
from designspace_nextgen import _read_rules  # type: ignore


def test_empty_conditionset() -> None:
    xml = """
    <designspace>
        <rules>
            <rule name="always_apply">
                <conditionset/>
                <sub name="a" with="a.alt"/>
            </rule>
        </rules>
    </designspace>
    """

    root = ElementTree.fromstring(xml)
    rules, rules_processing_last = _read_rules(root)

    assert not rules_processing_last
    assert rules == [
        Rule(
            name="always_apply",
            condition_sets=[ConditionSet(conditions={})],
            substitutions={"a": "a.alt"},
        )
    ]

    assert rules[0].evaluate({"Weight": 400}, {"a", "a.alt"}) == [("a", "a.alt")]


def test_stray_conditionset() -> None:
    xml = """
    <designspace>
        <rules>
            <rule name="always_apply">
                <condition name="Italic" minimum="0.1" maximum="1"/>
                <sub name="a" with="a.alt"/>
            </rule>
        </rules>
    </designspace>
    """

    root = ElementTree.fromstring(xml)
    rules, rules_processing_last = _read_rules(root)

    assert not rules_processing_last
    assert rules == [
        Rule(
            name="always_apply",
            condition_sets=[ConditionSet(conditions={"Italic": Range(0.1, 1)})],
            substitutions={"a": "a.alt"},
        )
    ]

    assert rules[0].evaluate({"Italic": 1}, {"a", "a.alt"}) == [("a", "a.alt")]


def test_parse_rules() -> None:
    xml = """
    <designspace>
        <rules processing="last">
            <rule name="BRACKET.CYR">
                <conditionset>
                    <condition name="A" minimum="0.1" maximum="1"/>
                </conditionset>
                <sub name="a" with="a.alt"/>
            </rule>
                <rule name="BRACKET.116.185">
                <conditionset>
                    <condition name="B" minimum="116" maximum="185"/>
                    <condition name="C" minimum="75" maximum="97.5"/>
                </conditionset>
                <sub name="cent" with="cent.alt"/>
                <sub name="dollar" with="dollar.alt"/>
            </rule>
        </rules>
    </designspace>
    """

    root = ElementTree.fromstring(xml)
    rules, rules_processing_last = _read_rules(root)

    assert rules_processing_last
    assert rules == [
        Rule(
            name="BRACKET.CYR",
            condition_sets=[ConditionSet(conditions={"A": Range(0.1, 1)})],
            substitutions={"a": "a.alt"},
        ),
        Rule(
            name="BRACKET.116.185",
            condition_sets=[
                ConditionSet(conditions={"B": Range(116, 185), "C": Range(75, 97.5)}),
            ],
            substitutions={"cent": "cent.alt", "dollar": "dollar.alt"},
        ),
    ]

    assert rules[0].evaluate({"C": 100, "A": 0}, {"a", "a.alt"}) == []
    assert rules[0].evaluate({"A": 0.1}, {"a", "a.alt"}) == [("a", "a.alt")]
    assert rules[0].evaluate({"b": 400, "A": 1}, {"a", "a.alt"}) == [("a", "a.alt")]

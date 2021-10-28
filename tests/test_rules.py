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
                    <condition name="Italic" minimum="0.1" maximum="1"/>
                </conditionset>
                <sub name="ghe.loclSRB" with="ghe.ital.loclSRB"/>
                <sub name="ghe.loclMKD" with="ghe.ital.loclMKD"/>
                <sub name="de.loclMKDSRB" with="de.ital.loclMKDSRB"/>
                <sub name="pe.loclMKDSRB" with="pe.ital.loclMKDSRB"/>
                <sub name="te.loclMKDSRB" with="te.ital.loclMKDSRB"/>
                <sub name="gje.loclMKD" with="gje.ital.loclMKD"/>
                <sub name="sha.loclMKDSRB" with="sha.ital.loclMKDSRB"/>
            </rule>
                <rule name="BRACKET.116.185">
                <conditionset>
                    <condition name="Weight" minimum="116" maximum="185"/>
                    <condition name="Width" minimum="75" maximum="97.5"/>
                </conditionset>
                <sub name="cent" with="cent.BRACKET.130"/>
                <sub name="dollar" with="dollar.BRACKET.130"/>
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
            condition_sets=[ConditionSet(conditions={"Italic": Range(0.1, 1)})],
            substitutions={
                "ghe.loclSRB": "ghe.ital.loclSRB",
                "ghe.loclMKD": "ghe.ital.loclMKD",
                "de.loclMKDSRB": "de.ital.loclMKDSRB",
                "pe.loclMKDSRB": "pe.ital.loclMKDSRB",
                "te.loclMKDSRB": "te.ital.loclMKDSRB",
                "gje.loclMKD": "gje.ital.loclMKD",
                "sha.loclMKDSRB": "sha.ital.loclMKDSRB",
            },
        ),
        Rule(
            name="BRACKET.116.185",
            condition_sets=[
                ConditionSet(
                    conditions={"Weight": Range(116, 185), "Width": Range(75, 97.5)}
                ),
            ],
            substitutions={"cent": "cent.BRACKET.130", "dollar": "dollar.BRACKET.130"},
        ),
    ]

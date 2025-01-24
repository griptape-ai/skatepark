from griptape.mixins.rule_mixin import RuleMixin
from griptape.rules import JsonSchemaRule, Rule, Ruleset
from griptape.structures import Agent
from griptape.tasks import PromptTask


class TestRuleMixin:
    def test_no_rules(self):
        assert RuleMixin()

    def test_rules(self):
        rule = Rule("foo")
        mixin = RuleMixin(rules=[rule])

        assert mixin.rules == [rule]

    def test_rulesets(self):
        ruleset1 = Ruleset("foo", [Rule("bar")])
        mixin = RuleMixin(rulesets=[ruleset1])

        assert mixin.rulesets == [ruleset1]
        ruleset2 = Ruleset("baz", [Rule("qux")])
        mixin.rulesets.append(ruleset2)
        assert mixin.rulesets == [ruleset1, ruleset2]

    def test_rules_and_rulesets(self):
        mixin = RuleMixin(rules=[Rule("foo")], rulesets=[Ruleset("bar", [Rule("baz")])])

        assert mixin.rules == [Rule("foo")]
        assert mixin.all_rulesets[0].name == "bar"
        assert mixin.all_rulesets[0].rules == [Rule("baz")]
        assert mixin.all_rulesets[1].name == "Default Ruleset"
        assert mixin.all_rulesets[1].rules == [Rule("foo")]

    def test_inherits_structure_rulesets(self):
        # Tests that a task using the mixin inherits rulesets from its structure.
        ruleset1 = Ruleset("foo", [Rule("foo rule")])
        ruleset2 = Ruleset("bar", [Rule("bar rule")])

        agent = Agent(rulesets=[ruleset1])
        task = PromptTask(rulesets=[ruleset2])
        agent.add_task(task)

        assert task.all_rulesets == [ruleset1, ruleset2]

    def test_to_dict(self):
        mixin = RuleMixin(
            rules=[
                Rule("foo"),
                JsonSchemaRule(
                    {
                        "type": "object",
                        "properties": {
                            "foo": {"type": "string"},
                        },
                        "required": ["foo"],
                    }
                ),
            ],
            rulesets=[Ruleset("bar", [Rule("baz")])],
        )

        assert mixin.to_dict() == {
            "rules": [
                {"type": "Rule", "value": "foo"},
                {
                    "type": "JsonSchemaRule",
                    "value": {
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                        "type": "object",
                    },
                },
            ],
            "rulesets": [
                {
                    "id": mixin.all_rulesets[0].id,
                    "meta": {},
                    "name": "bar",
                    "rules": [{"type": "Rule", "value": "baz"}],
                    "type": "Ruleset",
                },
            ],
            "type": "RuleMixin",
        }

    def test_from_dict(self):
        mixin = RuleMixin(
            rules=[
                Rule("foo"),
                JsonSchemaRule(
                    {
                        "type": "object",
                        "properties": {
                            "foo": {"type": "string"},
                        },
                        "required": ["foo"],
                    }
                ),
            ],
            rulesets=[Ruleset("bar", [Rule("baz")])],
        )

        new_mixin = RuleMixin.from_dict(mixin.to_dict())

        for idx, _ in enumerate(new_mixin.rulesets):
            rules = mixin.rulesets[idx].rules
            new_rules = new_mixin.rulesets[idx].rules
            for idx, _ in enumerate(rules):
                assert rules[idx].value == new_rules[idx].value
                assert rules[idx].meta == new_rules[idx].meta

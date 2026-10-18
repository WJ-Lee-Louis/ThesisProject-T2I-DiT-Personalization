"""Verify prompt examples obey the same span/deletion contract as live output."""
from itertools import combinations
import json
from pathlib import Path

import pytest

from orthogonalization_instantx_ip_adapter.semantic_components import COMPONENT_SYSTEM_PROMPT, SemanticLLM


def examples():
    lines = COMPONENT_SYSTEM_PROMPT.splitlines()
    return [(json.loads(lines[i - 1]), json.loads(line[3:]))
            for i, line in enumerate(lines) if line.startswith('=> ')]


@pytest.mark.parametrize('payload,response', examples())
def test_target_few_shots_partition_and_counterfactuals(payload, response):
    class Provider:
        def complete(self, **kwargs):
            assert kwargs['payload'] == payload
            return json.dumps(response)
    analysis = SemanticLLM(Provider(), max_retries=0).analyze_target(**payload)
    n = len(analysis.components)
    assert analysis.prompt_for_subset(range(n)) == payload['text']
    fixed = response['preserved_spans'] + response['neutral_spans']
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            result = analysis.prompt_for_subset(subset)
            assert all(span in result for span in fixed)


def test_shared_target_prompt_matches_runtime():
    root = Path(__file__).resolve().parents[1]
    assert COMPONENT_SYSTEM_PROMPT in (root/'docs/prompts/llm_modules_share.md').read_text(encoding='utf-8')

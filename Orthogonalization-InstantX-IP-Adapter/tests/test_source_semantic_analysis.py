"""Source selection, fixed context geometry, and persisted target binding."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import pytest

from orthogonalization_instantx_ip_adapter.semantic_components import (
    CAPTION_SYSTEM_PROMPT, COMPONENT_SYSTEM_PROMPT, ComponentAnalysis,
    IdentityConflictError, SemanticConditioning, SemanticLLM, prepare_semantic_conditioning,
)
from orthogonalization_instantx_ip_adapter.source_component_prompt import (
    SOURCE_EXAMPLES, SOURCE_COMPONENT_SYSTEM_PROMPT, SOURCE_COMPONENT_SCHEMA,
)
from orthogonalization_instantx_ip_adapter.component_directions import component_contrast_plan


class Provider:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(next(self.outputs))


def analyze(index=0, output=None):
    inp, expected = SOURCE_EXAMPLES[index]
    provider = Provider(expected if output is None else output)
    return SemanticLLM(provider, max_retries=0).analyze_source(**inp)


@pytest.mark.parametrize('index', range(len(SOURCE_EXAMPLES) - 1))
def test_every_few_shot_partition_and_every_subset(index):
    inp, out = SOURCE_EXAMPLES[index]
    analysis = analyze(index)
    assert analysis.prompt_for_subset(range(len(analysis.components))) == inp['source_caption']
    for size in range(len(analysis.components) + 1):
        for subset in combinations(range(len(analysis.components)), size):
            prompt = analysis.prompt_for_subset(subset)
            assert all(span in prompt for span in out['preserved_spans'] + out['retained_spans'] + out['neutral_spans'])
    for strategy in ['local', 'factorial']:
        conditioning = SemanticConditioning(ComponentAnalysis(inp['target_prompt'], inp['identity_term'], ()),
                                            analysis, 'source', strategy)
        assert len(component_contrast_plan(conditioning)) == len(out['components'])
        assert SemanticConditioning.from_dict(conditioning.to_dict()) == conditioning


def test_same_source_target_scope_changes_selection_only():
    location, accessory, agreement, recolor = [analyze(i) for i in range(4)]
    assert location.text == accessory.text == agreement.text == recolor.text
    assert location.components == accessory.components == agreement.components
    assert location.retained_spans == accessory.retained_spans == agreement.retained_spans
    assert len(recolor.components) == 4 and recolor.retained_spans == ()


def test_factorial_varies_only_removal_candidates():
    source = analyze()
    target = ComponentAnalysis(source.conditioned_on_target, source.identity_term, ())
    plan = component_contrast_plan(SemanticConditioning(target, source, 'source', 'factorial', 4))
    assert len(plan) == 2
    assert all(len(pairs) == 2 for _, pairs in plan)
    for _, pairs in plan:
        for pair in pairs:
            assert all('brown' in p and 'with white paws' in p for p in pair)
    assert plan[0][1][0] == ('a brown dog with white paws sitting', 'a brown dog with white paws')


@pytest.mark.parametrize('mutation', [
    lambda o: o.update(retained_spans=['purple']),  # Target/invented text is not a source span.
    lambda o: o.update(retained_spans=['brown', 'with white paws', 'dog']),
    lambda o: o.update(retained_spans=['with white paws', 'brown']),
    lambda o: o.update(retained_spans=['brown']),  # Coverage omission.
    lambda o: o.update(preserved_spans=['brown']),
    lambda o: o.update(components=[{'spans': ['brown']}, {'spans': ['sitting']}, {'spans': ['on grass']}]),
    lambda o: o.update(components=['sitting', 'on grass']),
    lambda o: o.update(neutral_spans=None),
    lambda o: o.update(extra='unsupported'),
])
def test_invalid_source_partition_rejected(mutation):
    out = deepcopy(SOURCE_EXAMPLES[0][1])
    mutation(out)
    with pytest.raises(ValueError):
        analyze(output=out)


def test_conflict_stops_without_retry_or_target_call():
    inp, out = SOURCE_EXAMPLES[-1]
    provider = Provider(out)
    with pytest.raises(IdentityConflictError, match='purple'):
        prepare_semantic_conditioning(llm=SemanticLLM(provider), **inp)
    assert len(provider.calls) == 1


def test_fabricated_conflict_evidence_is_invalid_output():
    inp, out = SOURCE_EXAMPLES[-1]
    out = deepcopy(out)
    out['conflicts'][0]['target_spans'] = ['not in the target']
    with pytest.raises(ValueError, match='verbatim'):
        SemanticLLM(Provider(out), max_retries=0).analyze_source(**inp)


@pytest.mark.parametrize('origin', ['source', 'source_target'])
def test_pipeline_routes_target_to_source_without_image(origin):
    inp, out = SOURCE_EXAMPLES[0]
    target_out = dict(identity_term='dog', preserved_spans=['a dog'], neutral_spans=[],
                      components=[{'spans': ['on the beach']}])
    provider = Provider(out, target_out)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), **inp, origin=origin)
    assert provider.calls[0]['payload'] == inp
    assert provider.calls[0]['system_prompt'] == SOURCE_COMPONENT_SYSTEM_PROMPT
    assert provider.calls[0]['schema'] == SOURCE_COMPONENT_SCHEMA
    assert provider.calls[0]['image_bytes'] is None
    assert len(provider.calls) == (1 if origin == 'source' else 2)
    if origin == 'source_target':
        assert provider.calls[1]['system_prompt'] == COMPONENT_SYSTEM_PROMPT
        assert provider.calls[1]['payload'] == {'text': inp['target_prompt'], 'identity_term': 'dog'}
    assert result.source.conditioned_on_target == inp['target_prompt']


def test_target_only_does_not_request_source_analysis():
    provider = Provider(dict(identity_term='dog', preserved_spans=['a dog'], neutral_spans=[], components=[]))
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term='dog', target_prompt='a dog', origin='target')
    assert len(provider.calls) == 1 and result.source is None


def test_retained_attributes_can_be_disjoint_and_zero_removal_is_valid():
    source = analyze(8)
    assert len(source.retained_spans) == 3
    conditioning = SemanticConditioning(ComponentAnalysis(source.conditioned_on_target, source.identity_term, ()), source, 'source')
    assert component_contrast_plan(conditioning) == ()
    assert source.prompt_for_subset([]) == source.text


def test_source_target_binding_and_role_rejected_on_reload():
    source = analyze()
    conditioning = SemanticConditioning(ComponentAnalysis(source.conditioned_on_target, 'dog', ()), source, 'source')
    payload = conditioning.to_dict()
    payload['target']['text'] = 'a dog on a sofa'
    with pytest.raises(ValueError, match='different target'):
        SemanticConditioning.from_dict(payload)
    with pytest.raises(ValueError, match='source retention policy'):
        SemanticConditioning(source, origin='target')


@pytest.mark.parametrize('field', ['retained_spans', 'conditioned_on_target', 'source_policy'])
def test_partial_source_metadata_not_silently_downgraded(field):
    source = analyze()
    payload = SemanticConditioning(ComponentAnalysis(source.conditioned_on_target, 'dog', ()), source, 'source').to_dict()
    del payload['source'][field]
    with pytest.raises(ValueError):
        SemanticConditioning.from_dict(payload)


def test_unversioned_legacy_source_keeps_legacy_semantics():
    payload = dict(target=dict(text='a dog on snow', identity_term='a dog', components=['on snow']),
                   source=dict(text='a dog running', identity_term='a dog', components=['running']),
                   origin='source_target', strategy='local', max_contexts=256)
    conditioning = SemanticConditioning.from_dict(payload)
    assert conditioning.source.retained_spans is None
    assert conditioning.source.conditioned_on_target is None
    assert len(component_contrast_plan(conditioning)) == 2
    assert conditioning.to_dict()['schema_version'] == 2


def test_runtime_caption_matches_reviewed_prompt_document():
    root = Path(__file__).resolve().parents[1]
    prompt = (root/'docs/prompts/module_a_caption_system_prompt.md').read_text(encoding='utf-8').split('```text\n', 1)[1].split('\n```', 1)[0]
    assert CAPTION_SYSTEM_PROMPT == prompt

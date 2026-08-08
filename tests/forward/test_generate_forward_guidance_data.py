import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "generate_forward_guidance_data.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_forward_guidance_data", SCRIPT_PATH,
)
guidance_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guidance_script)


def _record(terminal_tokens, sample_index):
    return {
        "product_tokens": [1, 4, 5],
        "terminal_tokens": terminal_tokens,
        "reward": 1.0,
        "source_index": 0,
        "sample_index": sample_index,
    }


def test_attach_beam_rewards_records_rank_and_canonical_cache() -> None:
    class FakeBeamScorer:
        def generate_batch(self, sources, **kwargs):
            assert sources == ["CO"]
            assert kwargs["beam_size"] == 2
            return [["CO", "C"]], torch.tensor([[0.0, -1.0]])

    records, metadata, report = guidance_script.attach_forward_rewards(
        [_record([1, 5, 4], 0), _record([1, 4, 5], 1)],
        {1: "<bos>", 4: "C", 5: "O"},
        FakeBeamScorer(),
        reward_mode="beam_reconstruction",
        batch_size=2,
        forward_beam_size=2,
        canonicalize_source=True,
    )
    assert [record["forward_beam_rank"] for record in records] == [1, 1]
    assert [record["reward"] for record in records] == [1.0, 1.0]
    assert all(record["validity_reward"] == 1.0 for record in records)
    assert metadata["forward_reward_unique_sources"] == 1
    assert metadata["forward_reward_generation_stats"] == {
        "input_count": 2,
        "valid_input_count": 2,
        "invalid_input_count": 0,
        "unique_valid_source_count": 1,
        "generated_source_count": 1,
        "pre_cached_source_count": 0,
        "deduplicated_input_count": 1,
    }
    assert metadata["forward_canonicalize_source"] is True
    assert report["reconstruction_rank_counts"] == {"0": 0, "1": 2, "2": 0}
    assert report["product_group_statistics"]["variable_reward_group_count"] == 0
    assert report["product_group_statistics"]["mean_unique_terminals"] == 2.0


def test_reward_parser_keeps_likelihood_as_backward_compatible_default() -> None:
    args = guidance_script.build_parser().parse_args([
        "--input_data", "input.pt",
        "--output_data", "output.pt",
        "--checkpoint", "forward.pt",
        "--vocab_file", "vocab.txt",
    ])
    assert args.reward_mode == "likelihood"
    assert args.forward_beam_size == 5
    assert args.canonicalize_source is False

from unittest.mock import patch

import pytest
import torch

from edit_flows.core.alignment import identity_align_xs_to_zs
from edit_flows.core.scheduler import CubicScheduler
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.sampling.euler_beam import sample_euler_beam
from edit_flows.training.trainer import prepare_batch, train_step
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN


def _product_memory_model() -> EditFlowsTransformer:
    return EditFlowsTransformer(
        vocab_size=24,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dim_feedforward=32,
        max_seq_len=16,
        dropout=0.0,
        attention_dropout=0.0,
        use_product_memory=True,
        product_memory_encoder_layers=1,
        product_memory_fusion_after_layers=[1, 2],
    )


class TestProductMemoryModel:
    def test_cached_memory_matches_raw_product_forward(self):
        torch.manual_seed(7)
        model = _product_memory_model().eval()
        x_t = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, 8, PAD_TOKEN],
        ])
        x_0 = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, 8, PAD_TOKEN],
        ])
        time = torch.tensor([[0.25], [0.75]])
        state_padding = x_t == PAD_TOKEN
        product_padding = x_0 == PAD_TOKEN

        raw_output = model(
            x_t,
            time,
            state_padding,
            product_tokens=x_0,
            product_padding_mask=product_padding,
        )
        cached_memory = model.encode_product(x_0, product_padding)
        cached_output = model(
            x_t,
            time,
            state_padding,
            product_memory=cached_memory,
            product_memory_padding_mask=product_padding,
        )

        for raw, cached in zip(raw_output, cached_output):
            assert torch.allclose(raw, cached, atol=1e-6, rtol=1e-6)

    def test_product_memory_requires_a_source_or_cache(self):
        model = _product_memory_model().eval()
        x_t = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
        with pytest.raises(ValueError, match="requires either cached"):
            model(x_t, torch.zeros(1, 1), x_t == PAD_TOKEN)

    def test_invalid_product_memory_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="encoder_layers >= 1"):
            EditFlowsTransformer(
                vocab_size=12,
                hidden_dim=16,
                num_layers=2,
                num_heads=4,
                use_product_memory=True,
            )
        with pytest.raises(ValueError, match="unique 1-based"):
            EditFlowsTransformer(
                vocab_size=12,
                hidden_dim=16,
                num_layers=2,
                num_heads=4,
                use_product_memory=True,
                product_memory_encoder_layers=1,
                product_memory_fusion_after_layers=[1, 1],
            )


class TestProductMemoryTraining:
    def test_prepare_batch_uses_unaligned_bos_prefixed_product(self):
        # This mimics a pre-aligned training row: GAP is an alignment aid,
        # never part of the product passed at inference.
        x_0 = torch.tensor([
            [4, GAP_TOKEN, 5, PAD_TOKEN],
            [6, 7, GAP_TOKEN, PAD_TOKEN],
        ])
        x_1 = torch.tensor([
            [4, 8, 5, PAD_TOKEN],
            [6, 7, 9, PAD_TOKEN],
        ])
        batch = prepare_batch(
            x_0,
            x_1,
            CubicScheduler(),
            identity_align_xs_to_zs,
            model_vocab_size=24,
            use_product_memory=True,
        )

        expected = torch.tensor([
            [BOS_TOKEN, 4, 5],
            [BOS_TOKEN, 6, 7],
        ])
        assert torch.equal(batch["product_tokens"], expected)
        assert torch.equal(
            batch["product_padding_mask"], expected == PAD_TOKEN,
        )

    def test_product_memory_parameters_receive_training_updates(self):
        torch.manual_seed(11)
        model = _product_memory_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_0 = torch.tensor([
            [4, GAP_TOKEN, 5, PAD_TOKEN],
            [6, 7, GAP_TOKEN, PAD_TOKEN],
        ])
        x_1 = torch.tensor([
            [4, 8, 5, PAD_TOKEN],
            [6, 7, 9, PAD_TOKEN],
        ])
        batch = prepare_batch(
            x_0,
            x_1,
            CubicScheduler(),
            identity_align_xs_to_zs,
            model_vocab_size=model.vocab_size,
            use_product_memory=True,
        )
        before = model.product_memory_encoder_layers[0].self_attn.in_proj_weight.detach().clone()

        metrics = train_step(model, batch, CubicScheduler(), optimizer)

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        after = model.product_memory_encoder_layers[0].self_attn.in_proj_weight
        assert not torch.equal(before, after)


class TestProductMemorySampling:
    def test_euler_encodes_product_only_once_when_cache_is_not_supplied(self):
        torch.manual_seed(19)
        model = _product_memory_model().eval()
        x_0 = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 6, PAD_TOKEN, PAD_TOKEN],
        ])

        with patch.object(
            model,
            "encode_product",
            wraps=model.encode_product,
        ) as encode_product:
            result, _ = sample_euler(
                model,
                x_0,
                CubicScheduler(),
                n_steps=3,
                max_seq_len=12,
            )

        assert result.shape[0] == x_0.shape[0]
        assert encode_product.call_count == 1

    def test_euler_reuses_a_supplied_product_cache(self):
        torch.manual_seed(23)
        model = _product_memory_model().eval()
        x_0 = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
        product_padding = x_0 == PAD_TOKEN
        cache = model.encode_product(x_0, product_padding)

        with patch.object(
            model,
            "encode_product",
            wraps=model.encode_product,
        ) as encode_product:
            sample_euler(
                model,
                x_0,
                CubicScheduler(),
                n_steps=3,
                max_seq_len=12,
                product_memory=cache,
                product_memory_padding_mask=product_padding,
            )

        assert encode_product.call_count == 0

    def test_euler_beam_reuses_original_product_memory_per_branch(self):
        """Branching must not re-encode or replace the immutable x_0 cache."""
        torch.manual_seed(29)
        model = _product_memory_model().eval()
        x_0 = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, PAD_TOKEN],
        ])
        kwargs = dict(
            n_branches=2,
            n_children=2,
            n_steps=3,
            max_seq_len=12,
            sample_seeds=[101, 202],
            score_mode="full_probability",
            changed_state_bonus=0.5,
            child_policy="stochastic_noop",
        )

        with patch.object(
            model, "encode_product", wraps=model.encode_product,
        ) as encode_product:
            implicit = sample_euler_beam(
                model, x_0, CubicScheduler(), **kwargs,
            )
        # One batched encode at sampler entry, never once per branch or step.
        assert encode_product.call_count == 1

        padding_mask = x_0 == PAD_TOKEN
        cache = model.encode_product(x_0, padding_mask)
        with patch.object(
            model, "encode_product", wraps=model.encode_product,
        ) as encode_product:
            explicit = sample_euler_beam(
                model,
                x_0,
                CubicScheduler(),
                product_memory=cache,
                product_memory_padding_mask=padding_mask,
                **kwargs,
            )
        assert encode_product.call_count == 0
        assert torch.equal(implicit, explicit)

    def test_euler_beam_product_memory_is_compatible_with_shared_forwards(self):
        """Deduplicated forwards must gather the matching immutable cache."""
        torch.manual_seed(31)
        model = _product_memory_model().eval()
        x_0 = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
        ])
        common = dict(
            n_branches=1,
            n_children=2,
            n_steps=3,
            max_seq_len=12,
            sample_seeds=[303, 404],
            score_mode="full_probability",
            changed_state_bonus=0.5,
            child_policy="stochastic_noop",
            profile_sample_group_size=2,
        )
        padding_mask = x_0 == PAD_TOKEN
        cache = model.encode_product(x_0, padding_mask)
        ordinary = sample_euler_beam(
            model,
            x_0,
            CubicScheduler(),
            product_memory=cache,
            product_memory_padding_mask=padding_mask,
            **common,
        )
        shared = sample_euler_beam(
            model,
            x_0,
            CubicScheduler(),
            product_memory=cache,
            product_memory_padding_mask=padding_mask,
            share_identical_forwards=True,
            **common,
        )
        assert torch.equal(ordinary, shared)

    def test_euler_beam_product_memory_accepts_first_event_center_bias(self):
        """The immutable product cache and first-event position bias coexist."""
        torch.manual_seed(37)
        model = _product_memory_model().eval()
        x_0 = torch.tensor([
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, PAD_TOKEN],
        ])
        padding_mask = x_0 == PAD_TOKEN
        cache = model.encode_product(x_0, padding_mask)
        scores = torch.zeros(2, 4, 3)
        scores[:, 1, :] = 1.0
        stats = {}

        with patch.object(
            model,
            "encode_product",
            wraps=model.encode_product,
        ) as encode_product:
            result = sample_euler_beam(
                model,
                x_0,
                CubicScheduler(),
                n_branches=1,
                n_children=2,
                n_steps=3,
                max_seq_len=12,
                product_memory=cache,
                product_memory_padding_mask=padding_mask,
                sample_seeds=[505, 606],
                score_mode="full_probability",
                changed_state_bonus=0.5,
                child_policy="stochastic_noop",
                profile_sample_group_size=2,
                first_event_position_scores=scores,
                first_event_position_bias_enabled=torch.tensor([True, True]),
                first_event_bias_max_multiplier=3.0,
                first_event_bias_stats=stats,
            )

        assert encode_product.call_count == 0
        assert result.shape[0] == x_0.shape[0]
        assert stats["summary_from_final_lineages"] is True
        assert stats["max_hazard_relative_error"] < 1e-6

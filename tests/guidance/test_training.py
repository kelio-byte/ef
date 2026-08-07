import torch

from edit_flows.guidance.data import collate_guidance_records, make_guidance_record
from edit_flows.guidance.model import ProductConditionedGuidance
from edit_flows.guidance.training import (
    evaluate_guidance_step,
    guidance_action_loss,
    train_guidance_step,
)
from edit_flows.utils.tokens import BOS_TOKEN


def _batch():
    records = [
        make_guidance_record(
            product_tokens=[BOS_TOKEN, 4, 5],
            state_tokens=[BOS_TOKEN, 4, 5],
            terminal_tokens=[BOS_TOKEN, 4, 6],
            time_step=0.5,
            reward=1.0,
            source_index=0,
            sample_index=0,
            time_index=1,
            sample_seed=1,
            coupling_seed=2,
        ),
        make_guidance_record(
            product_tokens=[BOS_TOKEN, 7],
            state_tokens=[BOS_TOKEN, 7],
            terminal_tokens=[BOS_TOKEN, 7],
            time_step=0.25,
            reward=0.0,
            source_index=1,
            sample_index=0,
            time_index=1,
            sample_seed=3,
            coupling_seed=4,
        ),
    ]
    return collate_guidance_records(records)


def test_guidance_action_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    model = ProductConditionedGuidance(
        vocab_size=16,
        hidden_dim=16,
        product_layers=1,
        state_layers=1,
        num_heads=4,
        dim_feedforward=32,
        max_seq_len=16,
        dropout=0.0,
        attention_dropout=0.0,
    )
    loss, metrics = guidance_action_loss(model, _batch())
    assert torch.isfinite(loss)
    assert metrics["selected_action_fraction"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["selected_guidance_mean"]))
    assert torch.isfinite(torch.tensor(metrics["reward_selected_guidance_corr"]))
    assert metrics["background_loss_weight"] == 0.01
    assert torch.isfinite(torch.tensor(metrics["loss_insert_selected"]))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_guidance_train_and_eval_steps_update_only_on_train():
    torch.manual_seed(1)
    model = ProductConditionedGuidance(
        vocab_size=16,
        hidden_dim=16,
        product_layers=1,
        state_layers=1,
        num_heads=4,
        dim_feedforward=32,
        max_seq_len=16,
        dropout=0.0,
        attention_dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    eval_metrics = evaluate_guidance_step(model, _batch())
    assert torch.isfinite(torch.tensor(eval_metrics["loss"]))
    assert all(torch.equal(a, b) for a, b in zip(
        before, model.parameters(), strict=True,
    ))
    train_metrics = train_guidance_step(model, _batch(), optimizer)
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert any(not torch.equal(a, b) for a, b in zip(
        before, model.parameters(), strict=True,
    ))

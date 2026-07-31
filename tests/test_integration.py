import torch
import torch.nn as nn
import torch.nn.functional as F
from edit_flows.core.scheduler import CubicScheduler
from edit_flows.core.coupling import EmptyCoupling, UniformCoupling
from edit_flows.core.alignment import opt_align_xs_to_zs
from edit_flows.training.trainer import prepare_batch, train_step
from edit_flows.sampling.euler import sample_euler
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


class SmallMLP(nn.Module):
    """Simple MLP model for integration testing without Transformer instability."""
    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rates_head = nn.Linear(hidden_dim, 3)
        self.ins_head = nn.Linear(hidden_dim, vocab_size)
        self.sub_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        b, l = tokens.shape
        if l == 0:
            return (
                torch.empty(b, 0, 3, device=tokens.device),
                torch.empty(b, 0, self.vocab_size, device=tokens.device),
                torch.empty(b, 0, self.vocab_size, device=tokens.device),
            )
        tok_emb = self.embed(tokens)
        t_emb = self.time_proj(time_step).unsqueeze(1).expand(-1, l, -1)
        x = self.mlp(tok_emb + t_emb)
        rates = F.softplus(self.rates_head(x))
        ins_probs = F.softmax(self.ins_head(x), dim=-1)
        sub_probs = F.softmax(self.sub_head(x), dim=-1)
        m = (~padding_mask).unsqueeze(-1).float()
        return rates * m, ins_probs * m, sub_probs * m


class TestEndToEnd:
    def test_full_training_loop(self):
        V = 16
        torch.manual_seed(42)
        model = SmallMLP(vocab_size=V + 3, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CubicScheduler()
        coupling = EmptyCoupling()

        x_1 = torch.tensor([
            [3, 4, 5, 6, PAD_TOKEN, PAD_TOKEN],
            [7, 8, 9, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [10, 11, 12, 13, 14, PAD_TOKEN],
            [15, 16, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ])

        losses = []
        for _ in range(30):
            x_0, x_1_batch = coupling.sample(x_1)
            batch = prepare_batch(x_0, x_1_batch, scheduler, opt_align_xs_to_zs, model_vocab_size=V + 3)
            metrics = train_step(model, batch, scheduler, optimizer)
            losses.append(metrics["loss"])
        assert not any(torch.isnan(torch.tensor(l)).item() for l in losses)

    def test_sampling_from_trained_model(self):
        V = 16
        model = EditFlowsTransformer(
            vocab_size=V + 3,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        scheduler = CubicScheduler()

        model.eval()
        x_0 = torch.empty((2, 0), dtype=torch.long)
        result, _ = sample_euler(
            model, x_0, scheduler,
            n_steps=10, max_seq_len=64,
        )
        assert result.shape[0] == 2
        assert result.shape[1] <= 64

    def test_model_output_signature(self):
        V = 16
        model = EditFlowsTransformer(
            vocab_size=V + 3,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        tokens = torch.tensor([
            [BOS_TOKEN, 3, 4, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 5, 6, 7, PAD_TOKEN],
        ])
        t = torch.tensor([[0.3], [0.7]])
        padding_mask = tokens == PAD_TOKEN

        log_rates, log_ins_probs, log_sub_probs = model(tokens, t, padding_mask)

        assert log_rates.shape == (2, 5, 3)
        assert log_ins_probs.shape == (2, 5, V + 3)
        assert log_sub_probs.shape == (2, 5, V + 3)
        non_pad = ~padding_mask
        assert (log_rates[non_pad] > -1e8).all()
        ins_probs = torch.exp(log_ins_probs)
        assert torch.allclose(ins_probs[non_pad].sum(dim=-1), torch.ones(non_pad.sum()))


class TestOverfit:
    def test_single_example_overfit(self):
        V = 16
        torch.manual_seed(42)
        model = SmallMLP(vocab_size=V + 3, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = CubicScheduler()
        coupling = EmptyCoupling()

        x_1 = torch.tensor([[3, 4, 5, PAD_TOKEN]])
        x_0, x_1_batch = coupling.sample(x_1)

        losses = []
        for _ in range(300):
            batch = prepare_batch(x_0, x_1_batch, scheduler, opt_align_xs_to_zs, model_vocab_size=V + 3)
            metrics = train_step(model, batch, scheduler, optimizer)
            losses.append(metrics["loss"])

        assert not torch.isnan(torch.tensor(losses[-1]))
        assert losses[-1] < losses[0] * 0.5

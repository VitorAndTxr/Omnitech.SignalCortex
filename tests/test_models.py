"""Tests for model forward passes — correct output shapes for all architectures."""

import pytest
import torch

from configs.config import ModelConfig
from models import build_model, BaseModel
from models.lstm import LSTMModel, Attention, build_lstm_encoder
from models.tcn import TCNModel, TemporalBlock, Chomp1d, build_tcn_encoder
from models.transformer import TransformerModel, SinusoidalPositionalEncoding, build_transformer_encoder
from models.multiscale import MultiScaleModel


NUM_FEATURES = 8
BATCH = 4
SEQ_LEN = 32


def _default_cfg(**kwargs) -> ModelConfig:
    cfg = ModelConfig()
    cfg.type = "multiscale"
    cfg.branch_encoder = "lstm"
    cfg.num_layers = 2
    cfg.dropout = 0.0  # deterministic in tests
    cfg.bidirectional = True
    cfg.use_attention = True
    cfg.kernel_size = 3
    cfg.num_channels = [16, 16]
    cfg.nhead = 4
    cfg.dim_feedforward = 64
    cfg.branch_hidden_sizes = [16, 8, 8]
    cfg.branch_window_sizes = [SEQ_LEN, SEQ_LEN // 2, SEQ_LEN // 4]
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _dummy_input():
    return torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    def test_output_shapes(self):
        hidden = 16
        attn = Attention(hidden)
        lstm_out = torch.randn(BATCH, SEQ_LEN, hidden)
        context, weights = attn(lstm_out)
        assert context.shape == (BATCH, hidden)
        assert weights.shape == (BATCH, SEQ_LEN, 1)

    def test_weights_sum_to_one(self):
        attn = Attention(16)
        lstm_out = torch.randn(BATCH, SEQ_LEN, 16)
        _, weights = attn(lstm_out)
        sums = weights.squeeze(-1).sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# Single-scale model classes (kept as building blocks, still instantiable)
# ---------------------------------------------------------------------------

class TestLSTMModel:
    def _lstm_cfg(self, **kwargs):
        cfg = ModelConfig()
        cfg.type = "multiscale"  # type field; LSTMModel uses hidden_size etc.
        cfg.hidden_size = 32
        cfg.num_layers = 2
        cfg.dropout = 0.0
        cfg.bidirectional = True
        cfg.use_attention = True
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_output_shape_with_attention(self):
        cfg = self._lstm_cfg(use_attention=True)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_output_shape_without_attention(self):
        cfg = self._lstm_cfg(use_attention=False)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_unidirectional(self):
        cfg = self._lstm_cfg(bidirectional=False, use_attention=True)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_single_layer_no_dropout(self):
        cfg = self._lstm_cfg(num_layers=1, dropout=0.1)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)


class TestTCNModel:
    def _tcn_cfg(self, **kwargs):
        cfg = ModelConfig()
        cfg.type = "multiscale"
        cfg.num_channels = [16, 16]
        cfg.kernel_size = 3
        cfg.dropout = 0.0
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_output_shape(self):
        cfg = self._tcn_cfg()
        model = TCNModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_single_block(self):
        cfg = self._tcn_cfg(num_channels=[32])
        model = TCNModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_chomp1d_removes_trailing(self):
        chomp = Chomp1d(5)
        x = torch.randn(2, 16, 30)
        out = chomp(x)
        assert out.shape == (2, 16, 25)

    def test_temporal_block_same_channels(self):
        block = TemporalBlock(16, 16, kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(2, 16, 20)
        out = block(x)
        assert out.shape == x.shape

    def test_temporal_block_different_channels(self):
        block = TemporalBlock(8, 16, kernel_size=3, dilation=2, dropout=0.0)
        x = torch.randn(2, 8, 20)
        out = block(x)
        assert out.shape == (2, 16, 20)


class TestTransformerModel:
    def _transformer_cfg(self, **kwargs):
        cfg = ModelConfig()
        cfg.type = "multiscale"
        cfg.hidden_size = 32
        cfg.num_layers = 2
        cfg.dropout = 0.0
        cfg.nhead = 4
        cfg.dim_feedforward = 64
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_output_shape(self):
        cfg = self._transformer_cfg()
        model = TransformerModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_positional_encoding_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=64, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, 32)
        out = pe(x)
        assert out.shape == x.shape

    def test_nhead_must_divide_hidden(self):
        cfg = self._transformer_cfg(hidden_size=32, nhead=3)
        with pytest.raises(Exception):
            TransformerModel(NUM_FEATURES, cfg)


# ---------------------------------------------------------------------------
# Encoder factories
# ---------------------------------------------------------------------------

class TestEncoderFactories:
    def _base_cfg(self):
        cfg = ModelConfig()
        cfg.num_layers = 1
        cfg.dropout = 0.0
        cfg.bidirectional = False
        cfg.use_attention = True
        cfg.num_channels = [16, 16]
        cfg.kernel_size = 3
        cfg.nhead = 4
        cfg.dim_feedforward = 32
        return cfg

    def test_lstm_encoder_output_shape(self):
        cfg = self._base_cfg()
        enc = build_lstm_encoder(NUM_FEATURES, 32, cfg)
        enc.eval()
        out = enc(torch.randn(BATCH, SEQ_LEN, NUM_FEATURES))
        assert out.shape == (BATCH, 32)

    def test_tcn_encoder_output_shape(self):
        cfg = self._base_cfg()
        enc = build_tcn_encoder(NUM_FEATURES, 32, cfg)
        enc.eval()
        out = enc(torch.randn(BATCH, SEQ_LEN, NUM_FEATURES))
        assert out.shape == (BATCH, 32)

    def test_transformer_encoder_output_shape(self):
        cfg = self._base_cfg()
        enc = build_transformer_encoder(NUM_FEATURES, 32, cfg)
        enc.eval()
        out = enc(torch.randn(BATCH, SEQ_LEN, NUM_FEATURES))
        assert out.shape == (BATCH, 32)


# ---------------------------------------------------------------------------
# MultiScaleModel
# ---------------------------------------------------------------------------

class TestMultiScaleModel:
    def test_output_shape_lstm_encoder(self):
        cfg = _default_cfg(branch_encoder="lstm")
        model = MultiScaleModel(NUM_FEATURES, cfg)
        model.eval()
        x_5m = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        x_15m = torch.randn(BATCH, SEQ_LEN // 2, NUM_FEATURES)
        x_1h = torch.randn(BATCH, SEQ_LEN // 4, NUM_FEATURES)
        out = model(x_5m, x_15m, x_1h)
        assert out.shape == (BATCH, 2)

    def test_output_shape_tcn_encoder(self):
        cfg = _default_cfg(branch_encoder="tcn")
        model = MultiScaleModel(NUM_FEATURES, cfg)
        model.eval()
        x_5m = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        x_15m = torch.randn(BATCH, SEQ_LEN // 2, NUM_FEATURES)
        x_1h = torch.randn(BATCH, SEQ_LEN // 4, NUM_FEATURES)
        out = model(x_5m, x_15m, x_1h)
        assert out.shape == (BATCH, 2)

    def test_output_shape_transformer_encoder(self):
        # branch_hidden_sizes must be divisible by nhead=4
        cfg = _default_cfg(branch_encoder="transformer", branch_hidden_sizes=[16, 8, 8], nhead=4)
        model = MultiScaleModel(NUM_FEATURES, cfg)
        model.eval()
        x_5m = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        x_15m = torch.randn(BATCH, SEQ_LEN // 2, NUM_FEATURES)
        x_1h = torch.randn(BATCH, SEQ_LEN // 4, NUM_FEATURES)
        out = model(x_5m, x_15m, x_1h)
        assert out.shape == (BATCH, 2)

    def test_unknown_encoder_raises(self):
        cfg = _default_cfg(branch_encoder="unknown")
        with pytest.raises(ValueError, match="Unknown branch_encoder"):
            MultiScaleModel(NUM_FEATURES, cfg)

    def test_switching_encoder_type_changes_branch_module_type(self):
        """Switching branch_encoder must produce branches of different module types, not the same."""
        cfg_lstm = _default_cfg(branch_encoder="lstm")
        cfg_tcn = _default_cfg(branch_encoder="tcn")
        model_lstm = MultiScaleModel(NUM_FEATURES, cfg_lstm)
        model_tcn = MultiScaleModel(NUM_FEATURES, cfg_tcn)

        # Branch module type should differ between encoders
        lstm_branch_type = type(model_lstm.branch_5m).__name__
        tcn_branch_type = type(model_tcn.branch_5m).__name__
        assert lstm_branch_type != tcn_branch_type, (
            f"Expected different branch types but both are {lstm_branch_type!r}"
        )

    def test_all_three_encoder_types_produce_same_output_shape(self):
        """All encoder types must produce identical output shape (BATCH, 2)."""
        x_5m = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        x_15m = torch.randn(BATCH, SEQ_LEN // 2, NUM_FEATURES)
        x_1h = torch.randn(BATCH, SEQ_LEN // 4, NUM_FEATURES)

        for enc_type in ("lstm", "tcn", "transformer"):
            extra = {"branch_hidden_sizes": [16, 8, 8], "nhead": 4} if enc_type == "transformer" else {}
            cfg = _default_cfg(branch_encoder=enc_type, **extra)
            model = MultiScaleModel(NUM_FEATURES, cfg)
            model.eval()
            with torch.no_grad():
                out = model(x_5m, x_15m, x_1h)
            assert out.shape == (BATCH, 2), f"encoder={enc_type!r} produced shape {out.shape}"


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_factory_returns_multiscale_model(self):
        cfg = _default_cfg()
        model = build_model(NUM_FEATURES, cfg)
        assert isinstance(model, MultiScaleModel)

    def test_non_multiscale_type_raises(self):
        cfg = _default_cfg(type="lstm")
        with pytest.raises(ValueError, match="Only 'multiscale'"):
            build_model(NUM_FEATURES, cfg)

    def test_unknown_type_also_raises(self):
        cfg = _default_cfg(type="unknown_arch")
        with pytest.raises(ValueError, match="Only 'multiscale'"):
            build_model(NUM_FEATURES, cfg)

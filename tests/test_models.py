"""Tests for model forward passes — correct output shapes for all architectures."""

import pytest
import torch

from configs.config import ModelConfig
from models import build_model, BaseModel
from models.lstm import LSTMModel, Attention
from models.tcn import TCNModel, TemporalBlock, Chomp1d
from models.transformer import TransformerModel, SinusoidalPositionalEncoding
from models.multiscale import MultiScaleModel


NUM_FEATURES = 8
BATCH = 4
SEQ_LEN = 32


def _default_cfg(**kwargs) -> ModelConfig:
    cfg = ModelConfig()
    cfg.hidden_size = 32
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


class TestLSTMModel:
    def test_output_shape_with_attention(self):
        cfg = _default_cfg(type="lstm", use_attention=True)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_output_shape_without_attention(self):
        cfg = _default_cfg(type="lstm", use_attention=False)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_unidirectional(self):
        cfg = _default_cfg(type="lstm", bidirectional=False, use_attention=True)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_single_layer_no_dropout(self):
        cfg = _default_cfg(type="lstm", num_layers=1, dropout=0.1)
        model = LSTMModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)


class TestTCNModel:
    def test_output_shape(self):
        cfg = _default_cfg(type="tcn")
        model = TCNModel(NUM_FEATURES, cfg)
        model.eval()
        out = model(_dummy_input())
        assert out.shape == (BATCH, 2)

    def test_single_block(self):
        cfg = _default_cfg(type="tcn", num_channels=[32])
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
    def test_output_shape(self):
        cfg = _default_cfg(type="transformer")
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
        # hidden_size=32, nhead=3 is not divisible — should raise during construction
        cfg = _default_cfg(type="transformer", hidden_size=32, nhead=3)
        with pytest.raises(Exception):
            TransformerModel(NUM_FEATURES, cfg)


class TestMultiScaleModel:
    def test_output_shape(self):
        cfg = _default_cfg(type="multiscale")
        model = MultiScaleModel(NUM_FEATURES, cfg)
        model.eval()
        x_5m = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        x_15m = torch.randn(BATCH, SEQ_LEN // 2, NUM_FEATURES)
        x_1h = torch.randn(BATCH, SEQ_LEN // 4, NUM_FEATURES)
        out = model(x_5m, x_15m, x_1h)
        assert out.shape == (BATCH, 2)


class TestBuildModel:
    @pytest.mark.parametrize("model_type", ["lstm", "tcn", "transformer"])
    def test_factory_creates_correct_type(self, model_type):
        cfg = _default_cfg(type=model_type)
        model = build_model(NUM_FEATURES, cfg)
        assert isinstance(model, BaseModel)

    def test_unknown_type_raises(self):
        cfg = _default_cfg(type="unknown_arch")
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model(NUM_FEATURES, cfg)

    def test_multiscale_factory(self):
        cfg = _default_cfg(type="multiscale")
        model = build_model(NUM_FEATURES, cfg)
        assert isinstance(model, MultiScaleModel)

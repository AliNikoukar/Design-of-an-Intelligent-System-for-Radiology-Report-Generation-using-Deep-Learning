"""
Baseline Architecture B: DenseNet-121 + Bahdanau Spatial Attention + LSTM Decoder.
Implements spatial attention weighting across the 7x7 grid of visual patch features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class BahdanauAttention(nn.Module):
    """
    Bahdanau additive attention mechanism:
    Score(encoder_features, decoder_hidden) = v^T * tanh(W_e * features + W_d * hidden)
    """

    def __init__(self, encoder_dim: int = 1024, hidden_dim: int = 512, attention_dim: int = 256):
        super().__init__()
        self.w_encoder = nn.Linear(encoder_dim, attention_dim)
        self.w_decoder = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: (Batch, NumPatches, EncoderDim)
            decoder_hidden: (Batch, HiddenDim)
        Returns:
            context: (Batch, EncoderDim)
            alpha: (Batch, NumPatches) attention weights
        """
        att1 = self.w_encoder(encoder_features)
        att2 = self.w_decoder(decoder_hidden).unsqueeze(1)
        scores = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class BaselineDenseNetAttentionLSTM(nn.Module):
    """
    Baseline B:
    - Visual Encoder: DenseNet-121 2D spatial feature map (49 patches of dim 1024)
    - Attention: Bahdanau additive spatial attention + Beta gating layer
    - Decoder: LSTMCell with concatenated [WordEmbedding, AttendedVisualContext]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 1024,
        attention_dim: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
        )
        self.beta_gate = nn.Linear(hidden_dim, encoder_dim)

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim, vocab_size)

    def forward(
        self,
        spatial_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (Batch, NumPatches, EncoderDim) e.g. (B, 49, 1024)
            input_ids: (Batch, SeqLen)
        Returns:
            logits: (Batch, SeqLen - 1, VocabSize)
        """
        mean_features = spatial_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_features))
        c = torch.tanh(self.init_c(mean_features))

        batch_size, seq_len = input_ids.shape
        inputs = input_ids[:, :-1]
        embeddings = self.embedding(inputs)

        outputs = []
        for t in range(seq_len - 1):
            context, _ = self.attention(spatial_features, h)
            gate = torch.sigmoid(self.beta_gate(h))
            gated_context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], gated_context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            out = self.fc_out(torch.cat([h, gated_context], dim=1))
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    @torch.inference_mode()
    def greedy_generate(
        self,
        spatial_features: torch.Tensor,
        max_len: int = 60,
    ) -> List[int]:
        self.eval()
        if spatial_features.dim() == 2:
            spatial_features = spatial_features.unsqueeze(0)

        mean_features = spatial_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_features))
        c = torch.tanh(self.init_c(mean_features))

        curr_token = torch.tensor([self.bos_token_id], device=spatial_features.device)
        generated = []

        for _ in range(max_len):
            context, _ = self.attention(spatial_features, h)
            gate = torch.sigmoid(self.beta_gate(h))
            gated_context = gate * context

            embed = self.embedding(curr_token)
            lstm_input = torch.cat([embed, gated_context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            out = self.fc_out(torch.cat([h, gated_context], dim=1))
            next_token = out.argmax(dim=-1).item()

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)
            curr_token = torch.tensor([next_token], device=spatial_features.device)

        return generated

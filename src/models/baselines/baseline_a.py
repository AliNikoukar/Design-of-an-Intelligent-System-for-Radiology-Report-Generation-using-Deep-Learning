"""
Baseline Architecture A: DenseNet-121 Visual Feature Extractor + LSTM Language Decoder.
Standard classic baseline for chest X-ray captioning (2018-2020 paradigm).
"""

import torch
import torch.nn as nn
from typing import Optional, List


class BaselineDenseNetLSTM(nn.Module):
    """
    Baseline A:
    - Visual Encoder: Pre-trained DenseNet-121 (global average pooled to 1024-dim vector)
    - State Initialization: Linear projection of visual vector into LSTM hidden/cell states
    - Language Decoder: Word embedding layer + single-layer LSTM + output linear projection
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 1024,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # State initialization layers from visual vector
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        # Decoder text layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: Pooled visual feature vector (Batch, EncoderDim)
            input_ids: Tokenized target sequence (Batch, SeqLen)
        Returns:
            logits: Next-token logits (Batch, SeqLen - 1, VocabSize)
        """
        # Initialize hidden and cell state from image representation
        h_0 = torch.tanh(self.init_h(visual_features)).unsqueeze(0)
        c_0 = torch.tanh(self.init_c(visual_features)).unsqueeze(0)

        # Autoregressive teacher forcing
        embeddings = self.embedding(input_ids[:, :-1])
        lstm_out, _ = self.lstm(embeddings, (h_0, c_0))
        logits = self.fc_out(lstm_out)
        return logits

    @torch.inference_mode()
    def greedy_generate(
        self,
        visual_features: torch.Tensor,
        max_len: int = 60,
    ) -> List[int]:
        """
        Greedy autoregressive token generation.
        """
        self.eval()
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)

        h = torch.tanh(self.init_h(visual_features)).unsqueeze(0)
        c = torch.tanh(self.init_c(visual_features)).unsqueeze(0)

        curr_token = torch.tensor([[self.bos_token_id]], device=visual_features.device)
        generated = []

        for _ in range(max_len):
            embed = self.embedding(curr_token)
            lstm_out, (h, c) = self.lstm(embed, (h, c))
            logits = self.fc_out(lstm_out.squeeze(1))
            next_token = logits.argmax(dim=-1).item()

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)
            curr_token = torch.tensor([[next_token]], device=visual_features.device)

        return generated

# encoding: utf-8
"""

@version: 1.0
@file: glyph_embedding

"""

from typing import List

import numpy as np
import torch
from torch import nn


class GlyphEmbedding(nn.Module):
    """Glyph2Image Embedding"""

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        print(font_npy_files)
        # import ipdb
        # ipdb.set_trace()
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)
        self.font_size = font_arrays[0].shape[-1]
        # N, C, H, W
        print(self.vocab_size, self.font_num, self.font_size)
        font_array = np.stack(font_arrays, axis=1)
        print(font_array.shape)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.font_size ** 2 * self.font_num,
            _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)

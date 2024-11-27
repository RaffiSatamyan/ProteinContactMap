from imports import *

# The proteins contact map prediction model, the base of model is Transformers


class ContactMapPredictionTransformer(nn.Module):
    def __init__(self, mini_emb_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=mini_emb_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                     dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        self.contact_prediction_head = nn.Linear(mini_emb_dim, 1)

    def forward(self, x, mask=None):

        # x: (batch_size, seq_len, embedding_dim) - ESM-2 embeddings
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Pairwise comparison and prediction
        batch_size, seq_len, _ = x.shape
        x = x.unsqueeze(1) - x.unsqueeze(2)  # Pairwise difference (batch_size, seq_len, seq_len, embedding_dim)

        x = self.contact_prediction_head(x).squeeze(-1)  # (batch_size, seq_len, seq_len)
        x = torch.sigmoid(x)  # Convert to probabilities

        return x


class ContactMapPredictionLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size_1=256, hidden_size_2=128, dropout_rate=0.3, linear_size=128):
        super().__init__()

        # Bidirectional GRU layers
        self.gru1 = nn.GRU(embedding_dim, hidden_size_1, bidirectional=False, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, bidirectional=False,
                           batch_first=True)  # Output from GRU1 is 512 because it's bidirectional

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Dense layer for contact map prediction (binary output per amino acid pair)
        self.dense = nn.Linear(linear_size, 1)  # Output: Binary contact map (1 for each amino acid pair)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # GRU Layers
        x, _ = self.gru1(x)  # Shape: [batch_size, seq_length, hidden_size_1]
        x, _ = self.gru2(x)  # Shape: [batch_size, seq_length, hidden_size_2]

        batch_size, seq_len, _ = x.shape
        x = x.unsqueeze(1) - x.unsqueeze(2)
        # Pairwise difference (batch_size, seq_len, seq_len, embedding_dim)
        x = self.dropout(x)

        x = self.dense(x).squeeze(-1)  # (batch_size, seq_len, seq_len)
        x = torch.sigmoid(x)  # Convert to probabilities

        return x

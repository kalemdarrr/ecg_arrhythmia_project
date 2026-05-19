import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    """
    Compresses and denoises the signal.
    In the ablation study, only the encoder part is used (as a feature extractor).
    Input shape: (Batch, 1, 250)
    """
    def __init__(self, input_len=250):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder (250 -> 125 -> 62)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder (62 -> 125 -> 250)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Since the signal is normalized between 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_features(self, x):
        return self.encoder(x)


class CNNFeatureExtractor(nn.Module):
    """
    1D CNN layer, learns local features (QRS etc.).
    If AE is used, it takes AE_out_channels (32). 
    If not, it directly takes the raw signal (1 channel).
    """
    def __init__(self, in_channels=1):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        return x


class BiGRULayer(nn.Module):
    """
    Bidirectional GRU to learn time-dependent correlations.
    """
    def __init__(self, input_size, hidden_size=64):
        super(BiGRULayer, self).__init__()
        # PyTorch GRU expected input: (batch, seq_len, input_size) if batch_first=True
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          bidirectional=True, batch_first=True)

    def forward(self, x):
        # x shape from CNN: (batch, channels, length)
        # GRU expects (batch, length, channels) -> channels will be input_size
        x = x.permute(0, 2, 1) 
        output, _ = self.gru(x)
        # output shape: (batch, seq_len, hidden_size * 2)
        return output


class AttentionLayer(nn.Module):
    """
    Applies attention over time steps (seq_len).
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        weights = self.attention_weights(x) # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1) # normalize over seq_len
        
        # Apply weights
        context_vector = torch.sum(weights * x, dim=1) # (batch, hidden_size)
        return context_vector


class ECGModel(nn.Module):
    """
    Main model combining all blocks, configurable for the Ablation Study.
    Ablation Flags:
    - use_ae: True/False (Use Autoencoder?)
    - use_gru: True/False (Use BiGRU?)
    - use_attention: True/False (Use Attention?)
    """
    def __init__(self, num_classes=5, use_ae=False, use_gru=False, use_attention=False, input_len=250):
        super(ECGModel, self).__init__()
        self.use_ae = use_ae
        self.use_gru = use_gru
        self.use_attention = use_attention
        
        # 1. Denoising Autoencoder
        if self.use_ae:
            self.ae = DenoisingAutoencoder(input_len)
            cnn_in_channels = 32 # AE encoder output channels
            seq_len = input_len // 4 # AE stride reduces length by 4 (250 -> 62)
        else:
            cnn_in_channels = 1
            seq_len = input_len
            
        # 2. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(in_channels=cnn_in_channels)
        seq_len = seq_len // 4 # CNN uses 2 MaxPool1d(2) -> reduces length by 4
        cnn_out_channels = 128
        
        # 3. BiGRU
        if self.use_gru:
            self.gru = BiGRULayer(input_size=cnn_out_channels, hidden_size=64)
            gru_out_channels = 128 # 64 * 2 (Bidirectional)
        else:
            gru_out_channels = cnn_out_channels
            
        # 4. Attention
        if self.use_attention:
            assert self.use_gru, "Attention is generally applied to RNN/GRU output!"
            self.attention = AttentionLayer(hidden_size=gru_out_channels)
            dense_input = gru_out_channels
        else:
            # Flatten layer if no attention is used
            dense_input = gru_out_channels * seq_len
            
        # 5. Dense Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dense_input, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, 250)
        
        if self.use_ae:
            # AE is pre-trained, we freeze it or fine-tune. Here we just use its features.
            x = self.ae.get_features(x)
            
        x = self.cnn(x) # Output: (Batch, channels, length)
        
        if self.use_gru:
            x = self.gru(x) # Output: (Batch, length, channels)
        else:
            # Prepare for flatten if no GRU/Attention
            x = x.permute(0, 2, 1) # (Batch, length, channels)
            
        if self.use_attention:
            x = self.attention(x) # Output: (Batch, channels)
        else:
            # Flatten (Batch, length * channels)
            x = x.contiguous().view(x.size(0), -1)
            
        out = self.classifier(x) # Output: (Batch, num_classes)
        return out

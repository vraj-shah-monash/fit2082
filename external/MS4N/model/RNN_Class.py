import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, config, num_classes=None, rnn_type='RNN', num_layers=2, bidirectional=False):
        """
        Vanilla RNN adapted for your classification framework
        
        Args:
            config (dict): Configuration dictionary containing 'Data_shape' and 'num_labels'
            num_classes (int): Number of output classes
            rnn_type (str): Type of RNN ('RNN', 'LSTM', 'GRU')
            num_layers (int): Number of recurrent layers
            bidirectional (bool): Whether to use bidirectional RNN
        """
        super(VanillaRNN, self).__init__()
        
        # Extract from config to match your framework
        self.input_channels = config['Data_shape'][1]
        self.seq_len = config['Data_shape'][2]
        self.num_classes = num_classes or config['num_labels']
        
        # Model parameters from config
        self.hidden_size = getattr(config, 'd_model', 64)
        self.num_layers = getattr(config, 'rnn_layers', num_layers)
        self.dropout = getattr(config, 'dropout', 0.1)
        self.bidirectional = getattr(config, 'bidirectional', bidirectional)
        self.rnn_type = getattr(config, 'rnn_type', rnn_type)
        
        #print(f"Initializing {self.rnn_type}: {self.input_channels} channels, {self.seq_len} seq_len, {self.num_classes} classes")
        #print(f"Hidden size: {self.hidden_size}, Layers: {self.num_layers}, Bidirectional: {self.bidirectional}")
        
        # Input projection to convert channels to hidden_size
        self.input_projection = nn.Linear(self.input_channels, self.hidden_size)
        
        # RNN layer
        if self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                nonlinearity='tanh'  # or 'relu'
            )
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        elif self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}. Choose from 'RNN', 'LSTM', 'GRU'")
        
        # Calculate output size after RNN
        rnn_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Post-RNN processing
        self.post_rnn_dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(rnn_output_size)
        
        # Classification head with multiple pooling strategies
        self.use_attention_pooling = getattr(config, 'use_attention_pooling', False)
        
        if self.use_attention_pooling:
            # Attention-based pooling
            self.attention_weights = nn.Linear(rnn_output_size, 1)
            self.classifier = nn.Sequential(
                nn.Linear(rnn_output_size, rnn_output_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(rnn_output_size // 2, self.num_classes)
            )
        else:
            # Simple pooling strategies
            self.pooling_strategy = getattr(config, 'pooling_strategy', 'last')  # 'last', 'mean', 'max'
            
            if self.pooling_strategy == 'adaptive':
                self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(rnn_output_size // 1, rnn_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(rnn_output_size // 2, self.num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(rnn_output_size // 1, rnn_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(rnn_output_size // 2, self.num_classes)
                )
    
    def attention_pooling(self, rnn_output):
        """
        Attention-based pooling over sequence dimension
        Args:
            rnn_output: [batch_size, seq_len, hidden_size]
        Returns:
            pooled: [batch_size, hidden_size]
        """
        # Compute attention weights
        attention_scores = self.attention_weights(rnn_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        pooled = torch.sum(rnn_output * attention_weights, dim=1)  # [batch_size, hidden_size]
        return pooled
    
    def forward(self, x):
        """
        Forward pass adapted for your framework
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Sequence_Length)
            
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, num_classes)
        """
        batch_size = x.size(0)
        
        # Convert from your format (B, C, L) to (B, L, C)
        x = x.permute(0, 2, 1)  # (B, L, C)
        
        # Project input channels to hidden_size
        x = self.input_projection(x)  # (B, L, hidden_size)
        
        # Initialize hidden state if needed
        if self.rnn_type.upper() == 'LSTM':
            # LSTM needs both hidden and cell states
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
            rnn_output, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            # RNN and GRU only need hidden state
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
            rnn_output, hn = self.rnn(x, h0)
        
        # rnn_output: [batch_size, seq_len, hidden_size * num_directions]
        rnn_output = self.post_rnn_dropout(rnn_output)
        rnn_output = self.layer_norm(rnn_output)
        
        # Pooling strategy to get sequence representation
        if self.use_attention_pooling:
            pooled = self.attention_pooling(rnn_output)
        elif self.pooling_strategy == 'last':
            if self.bidirectional:
                # For bidirectional, take last forward + first backward
                forward_last = rnn_output[:, -1, :self.hidden_size]
                backward_first = rnn_output[:, 0, self.hidden_size:]
                pooled = torch.cat([forward_last, backward_first], dim=1)
            else:
                pooled = rnn_output[:, -1, :]  # Take last timestep
        elif self.pooling_strategy == 'mean':
            pooled = torch.mean(rnn_output, dim=1)  # Average over sequence
        elif self.pooling_strategy == 'max':
            pooled, _ = torch.max(rnn_output, dim=1)  # Max over sequence
        elif self.pooling_strategy == 'adaptive':
            # Use adaptive pooling
            rnn_output = rnn_output.permute(0, 2, 1)  # (B, hidden_size, seq_len)
            pooled = self.adaptive_pool(rnn_output)  # (B, hidden_size, 1)
            pooled = pooled.squeeze(-1)  # (B, hidden_size)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Classification
        output = self.classifier(pooled)
        
        return output


# Additional variants for your repository
class SimpleRNN(VanillaRNN):
    """Simple RNN variant"""
    def __init__(self, config, num_classes=None):
        super().__init__(config, num_classes, rnn_type='RNN', num_layers=1, bidirectional=False)

class SimpleLSTM(VanillaRNN):
    """Simple LSTM variant"""
    def __init__(self, config, num_classes=None):
        super().__init__(config, num_classes, rnn_type='LSTM', num_layers=2, bidirectional=False)

class SimpleGRU(VanillaRNN):
    """Simple GRU variant"""
    def __init__(self, config, num_classes=None):
        super().__init__(config, num_classes, rnn_type='GRU', num_layers=2, bidirectional=False)

class BiLSTM(VanillaRNN):
    """Bidirectional LSTM variant"""
    def __init__(self, config, num_classes=None):
        super().__init__(config, num_classes, rnn_type='LSTM', num_layers=2, bidirectional=True)
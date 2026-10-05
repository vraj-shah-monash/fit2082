

import math
import torch
import torch.nn as nn
from einops import repeat


# ============================================================
# 1. DropoutNd  (channel-last: channel dim is the LAST dim)
# ============================================================

class DropoutNd(nn.Module):
    """Drops entire feature channels. Assumes channel-last input (B, *, C):
    the mask is shared across every axis except batch and channel, so a
    dropped channel is dropped consistently across the whole sequence --
    matching the original channel-first semantics, just on the new layout.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = list(x.shape)
        mask_shape = [shape[0]] + [1] * (len(shape) - 2) + [shape[-1]]
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=x.device, dtype=x.dtype))
        return x * mask / (1 - self.p)


# ============================================================
# 2. MS4NKernel or S4DKernel
# ============================================================

class MS4NKernel(nn.Module):
    """MS4N's kernel, built on the standard S4D (diagonal state space) formulation.
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.B = nn.Parameter(torch.ones(H, N // 2))

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1) * self.B
        A_bar = torch.exp(dtA)
        B_bar = ((A_bar - 1.0) / A)

        k = torch.arange(L, device=A.device)
        A_pow = A_bar.unsqueeze(-1) ** k.unsqueeze(0).unsqueeze(0)
        CB = C * B_bar
        
        # produced directly by einsum with no extra transpose() call.
        K = torch.einsum('hn,hnl->lh', CB, A_pow)  # (L, H)
        K = 2 * K.real
        return K

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim
# ============================================================
# 3. MS4NLayer  (channel-last throughout: input/output are (B, L, H))
# ============================================================

class MS4NLayer(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state

        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = MS4NKernel(self.h, N=self.n, **kernel_args)

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        
        self.output_linear_val = nn.Linear(self.h, self.h)
        self.output_linear_gate = nn.Linear(self.h, self.h)

    def forward(self, u, **kwargs):
        # u: (B, L, H) -- channel-last, no transpose needed.
        B, L, H = u.shape

        # Step 1: Transform to frequency domain (transform axis = dim 1 = L)
        k = self.kernel(L=L)                          # (L, H)
        k_f = torch.fft.rfft(k, n=2 * L, dim=0)        # (Lf, H)
        u_f = torch.fft.rfft(u, n=2 * L, dim=1)        # (B, Lf, H)

        # Step 2: Convolution in frequency domain (K  u)
        conv_f = k_f * u_f                             

        # Step 3: Transform back to time domain
        y_conv = torch.fft.irfft(conv_f, n=2 * L, dim=1)[:, :L, :]   # (B, L, H)

        # Step 4: Add feedthrough D*u in time domain.
        y = y_conv + u * self.D

        # Step 5: Non-linearity, regularization, and channel mixing
        y = self.dropout(self.activation(y))
        value = self.output_linear_val(y)
        gate = self.output_linear_gate(y)
        y = value * torch.sigmoid(gate)    # (B, L, H) -> (B, L, H)

        return y


# ============================================================
# 4. MS4NBlock  
# ============================================================

class MS4NBlock(nn.Module):
    """Wrapper around MS4NLayer with projection."""
    def __init__(self, input_dim=3, d_model=64, d_state=64, dropout=0.0):
        super().__init__()
        self.project_in = nn.Linear(input_dim, d_model)
        self.ms4n = MS4NLayer(d_model=d_model, d_state=d_state, dropout=dropout)

    def forward(self, x):
        # x: (B, L, input_dim) -> (B, L, d_model) -> (B, L, d_model)
        x = self.project_in(x)
        x = self.ms4n(x)
        return x


# ============================================================
# 5. MS4NStackedLayers
# ============================================================

class MS4NStackedLayers(nn.Module):
    """Stack of multiple MS4N layers with residual connections."""
    def __init__(self, input_dim=3, d_model=64, d_state=64, dropout=0.0,
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # First layer: projects input_dim -> d_model
        self.layers = nn.ModuleList([
            MS4NBlock(input_dim, d_model, d_state, dropout)
        ])

        # Subsequent layers: d_model -> d_model
        for _ in range(num_layers - 1):
            self.layers.append(
                MS4NBlock(d_model, d_model, d_state, dropout)
            )

        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            if i == 0:
                x = layer(x)
            else:
                x = x + layer(x)
            x = norm(x)
        return x


# ============================================================
# 6. MS4NClassifier
# ============================================================

class MS4NClassifier(nn.Module):
    """
    Full MS4N classifier with multiple layers.
    """

    def __init__(self,
                 config: dict,
                 num_classes: int = None,
                 num_layers: int = None,
                 d_model: int = None,
                 d_state: int = None):

        super().__init__()

        # ----------------------------------------------------
        # Extract data shape
        self.input_dim = config['Data_shape'][1]   # C
        self.seq_len = config['Data_shape'][2]     # L
        self.num_classes = num_classes or config['num_labels']

        # ----------------------------------------------------
        # Hyperparameters
        self.d_model = d_model or config.get('d_model', 64)
        self.d_state = d_state or config.get('d_state', 64)
        self.num_layers = num_layers or config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0.1)

        # ----------------------------------------------------
        # MS4N Layer Stack
        self.ms4n_stack = MS4NStackedLayers(
            input_dim=self.input_dim,
            d_model=self.d_model,
            d_state=self.d_state,
            dropout=self.dropout,
            num_layers=self.num_layers
        )

        # ----------------------------------------------------
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

    def forward(self, x):

        if x.shape[1] == self.input_dim and x.shape[2] == self.seq_len:
            x = x.permute(0, 2, 1)

        x = self.ms4n_stack(x)              # (B, L, d_model)

        # Global average pool over the sequence dim, and the Classification Head
        x = x.mean(dim=1)                  # (B, d_model)
        logits = self.classifier(x)        # (B, num_classes)

        return logits
"""
cnn_model.py

The main deep learning model for audio deepfake detection.

Architecture: CNN (Convolutional Neural Network) on mel spectrograms.

Why CNN instead of a plain neural network?
  A plain network treats every pixel as independent.
  A CNN uses small filters that slide across the image,
  detecting local patterns (edges, stripes, textures) regardless
  of where they appear. This is exactly what we need — the GAN
  artifact stripes can appear at any frequency or time position.

Input:  tensor of shape (batch_size, 1, 128, 128)
        → batch_size: how many spectrograms processed at once
        → 1: one colour channel (grayscale spectrogram)
        → 128, 128: height and width in pixels

Output: tensor of shape (batch_size, 1)
        → a number 0.0–1.0 per sample
        → close to 0 = model thinks it's REAL
        → close to 1 = model thinks it's FAKE
"""

import torch
import torch.nn as nn
from pathlib import Path


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv → BatchNorm → ReLU → Pool.

    We define this as its own class to keep the main model clean.
    All 4 blocks follow exactly this same pattern.

    Parameters:
        in_channels:  how many feature maps come in (1, 32, 64, or 128)
        out_channels: how many feature maps come out (32, 64, 128, or 256)
        pool:         'max' uses MaxPool2d, 'avg' uses AdaptiveAvgPool2d(4,4)
    """
    def __init__(self, in_channels: int, out_channels: int, pool: str = 'max'):
        super().__init__()

        layers = [
            # Conv2d: the main learning layer
            # kernel_size=3  → 3×3 filter
            # padding=1      → adds 1 pixel border so output is same size as input
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            # BatchNorm2d: normalises outputs so training is stable
            # It tracks the mean and variance of each feature map
            # and scales them to have mean≈0, std≈1
            nn.BatchNorm2d(out_channels),

            # ReLU: activation function — sets negatives to 0
            # inplace=True means it modifies the tensor in memory (saves RAM)
            nn.ReLU(inplace=True),
        ]

        if pool == 'max':
            # MaxPool2d(2, 2): look at 2×2 patches, keep only the maximum
            # Effect: halves the spatial size (128→64→32→16)
            layers.append(nn.MaxPool2d(2, 2))
        elif pool == 'avg':
            # AdaptiveAvgPool2d((4, 4)): no matter what size comes in,
            # output is always 4×4. Averages over whatever area needed.
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNN(nn.Module):
    """
    4-block CNN for binary audio deepfake classification.

    Data flow (batch_size=32 example):
        Input:  (32, 1, 128, 128)   ← 32 spectrograms, 1 channel, 128×128
        Block1: (32, 32, 64, 64)    ← 32 feature maps, halved size
        Block2: (32, 64, 32, 32)    ← 64 feature maps, halved again
        Block3: (32, 128, 16, 16)   ← 128 feature maps
        Block4: (32, 256, 4, 4)     ← 256 feature maps, forced to 4×4
        Flatten:(32, 4096)          ← 256 × 4 × 4 = 4096 numbers per sample
        Dense:  (32, 512)
        Dense:  (32, 128)
        Dense:  (32, 1)             ← one probability per sample
        Sigmoid:(32, 1)             ← squeezes to [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ── Convolutional feature extractor ───────────────────────────────
        self.block1 = ConvBlock(in_channels=1,   out_channels=32,  pool='max')
        self.block2 = ConvBlock(in_channels=32,  out_channels=64,  pool='max')
        self.block3 = ConvBlock(in_channels=64,  out_channels=128, pool='max')
        self.block4 = ConvBlock(in_channels=128, out_channels=256, pool='avg')

        # ── Classification head ───────────────────────────────────────────
        # After block4: tensor is (batch, 256, 4, 4)
        # After flatten: (batch, 256*4*4) = (batch, 4096)
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # Layer 1: 4096 → 512
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            # Dropout(0.5): randomly zeros 50% of neurons during training
            # This FORCES the network to learn redundant representations
            # Result: much better generalisation on unseen data
            nn.Dropout(p=0.5),

            # Layer 2: 512 → 128
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # Output layer: 128 → 1
            nn.Linear(128, 1),

            # Sigmoid: squeezes any number to range (0, 1)
            # Output = P(fake) — probability the audio is AI-generated
            nn.Sigmoid()
        )

        # Initialise weights using Kaiming He initialisation
        # This gives Conv layers a good starting point vs random
        self._init_weights()

    def _init_weights(self):
        """Better weight initialisation speeds up training convergence."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming He init: designed for layers followed by ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass — how data flows through the network.

        PyTorch calls this automatically when you do model(input).
        You never call forward() directly.
        """
        x = self.block1(x)   # (B, 32, 64, 64)
        x = self.block2(x)   # (B, 64, 32, 32)
        x = self.block3(x)   # (B, 128, 16, 16)
        x = self.block4(x)   # (B, 256, 4, 4)
        x = self.classifier(x)  # (B, 1)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns prediction probabilities with no gradient tracking.
        Use this at inference time (not during training).
        """
        self.eval()   # switch to eval mode (disables Dropout)
        with torch.no_grad():   # don't track gradients (saves memory)
            return self.forward(x)


# ── Utility: count trainable parameters ───────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Test when run directly ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing AudioCNN architecture...")

    model = AudioCNN()

    # Create a fake batch: 4 spectrograms of size 128×128
    # This simulates what the DataLoader will feed during training
    dummy_input = torch.randn(4, 1, 128, 128)
    print(f"\nInput shape:  {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}    (should be [4, 1])")
    print(f"Output values: {output.detach().numpy().flatten()}")
    print(f"  (all values should be between 0 and 1)")

    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"  (~{total_params/1e6:.1f}M parameters)")

    print("\n✓ CNN architecture is correct!")
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D Convolutional Embedding Layer
class ConvEmbedding(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, kernel_size=3, stride=1, padding=1):
        super(ConvEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim//2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(output_dim//2, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        # x shape: (B, N, 3)
        B, N, _ = x.shape
        
        # Reshape x for 1D convolution (B, 3, N)
        x = x.permute(0, 2, 1)  # x shape: (B, 3, N)
        
        # Apply Conv1d layers
        x = F.relu(self.conv1(x))  # x shape: (B, output_dim, N)
        x = F.relu(self.conv2(x))  # x shape: (B, output_dim, N)
        
        # Reshape back to (B, N, output_dim)
        x = x.permute(0, 2, 1)  # x shape: (B, N, output_dim)
        
        return x

# Transformer-based Model for Point Cloud
class PointCloudTransformer(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, n_heads=4, ff_dim=256, num_layers=4):
        super(PointCloudTransformer, self).__init__()
        
        # 1D Convolutional Embedding Layer
        self.embedding = ConvEmbedding(input_dim, output_dim)
        
        # Transformer Layer (using nn.TransformerEncoder)
   
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        
        # Embedding step (1D Conv)
        x = self.embedding(x)  # x: (B, N, output_dim)
     
        # Reshape for transformer (Seq, Batch, Feature)
        # x = x.permute(1, 0, 2)  # x: (N, B, output_dim)
        
        # Pass through Transformer encoder
        x = self.transformer(x)  # x: (N, B, output_dim)
        print(x.shape)
        # Apply average pooling (B, output_dim)
        # x = x.permute(1, 0, 2)  # x: (B, N, output_dim)
        x = x.mean(dim=1)  # x: (B, output_dim)
        
        return x

# Example usage
if __name__ == "__main__":
    B, N = 32, 1024  # Batch size, number of points
    input_data = torch.randn(B, N, 3)  # Example point cloud data (B, N, 3)
    
    model = PointCloudTransformer(input_dim=3, output_dim=128, n_heads=4, ff_dim=256, num_layers=4)
    
    output = model(input_data)
    print(f"Output shape: {output.shape}")  # Expected output: (B, output_dim)

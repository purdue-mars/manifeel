import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvPoolingHead(nn.Module):
    def __init__(self, input_channels, 
                 conv1_out_channels=128, 
                 conv2_out_channels=256, 
                 conv3_out_channels=512,
                 kernel_size=3, padding=1, use_se=False):
        super(ConvPoolingHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=conv1_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln1 = nn.LayerNorm(conv1_out_channels) 
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, 
                               out_channels=conv2_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln2 = nn.LayerNorm(conv2_out_channels)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, 
                               out_channels=conv3_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln3 = nn.LayerNorm(conv3_out_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if use_se:
            self.se1 = SEBlock(conv1_out_channels)
            self.se2 = SEBlock(conv2_out_channels)
        self.use_se = use_se
        
    def forward(self, x):

        x = self.conv1(x)                        
        x = x.permute(0, 2, 3, 1)                  
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se1(x)
        x = self.conv2(x)                         
        x = x.permute(0, 2, 3, 1)                   
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se2(x)
        x = self.conv3(x)                          
        x = x.permute(0, 2, 3, 1)                    
        x = self.ln3(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)                         
        x = x.view(x.size(0), -1)                    
        return x
    
class MlpHead(nn.Module):
    def __init__(self, input_dim, hidden1_dim=512, hidden2_dim=512, output_dim=4, dropout_rate=0.0):
        super(MlpHead, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp_layers(x)

from escnn import gspaces, nn
import torch

class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize,
            ),
            nn.ReLU(feat_type_hid, inplace=True),
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_hid,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquivariantResEncoder120Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)

        n1, n2, n3, n4, n5 = n_out // 16, n_out // 8, n_out // 4, n_out // 2, n_out

        self.conv = torch.nn.Sequential(
            # 120x120
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n1 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 116x116
            nn.ReLU(nn.FieldType(self.group, n1 * [self.group.regular_repr]), inplace=True),

            EquiResBlock(self.group, n1, n1, initialize=initialize),
            EquiResBlock(self.group, n1, n1, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n1 * [self.group.regular_repr]), 2),  # 58x58

            EquiResBlock(self.group, n1, n2, initialize=initialize),
            EquiResBlock(self.group, n2, n2, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n2 * [self.group.regular_repr]), 2),  # 29x29

            EquiResBlock(self.group, n2, n3, initialize=initialize),
            EquiResBlock(self.group, n3, n3, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n3 * [self.group.regular_repr]), 3),  # 9x9

            EquiResBlock(self.group, n3, n4, initialize=initialize),
            EquiResBlock(self.group, n4, n4, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n4 * [self.group.regular_repr]), 3),  # 3x3

            nn.R2Conv(
                nn.FieldType(self.group, n4 * [self.group.regular_repr]),
                nn.FieldType(self.group, n5 * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n5 * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        if isinstance(x, torch.Tensor):
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

class EquivariantResEncoder230Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)

        n1, n2, n3, n4, n5 = n_out // 16, n_out // 8, n_out // 4, n_out // 2, n_out

        self.conv = torch.nn.Sequential(
            # 230x230
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n1 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 226x226
            nn.ReLU(nn.FieldType(self.group, n1 * [self.group.regular_repr]), inplace=True),

            EquiResBlock(self.group, n1, n1, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n1 * [self.group.regular_repr]), 2),  # 113x113

            EquiResBlock(self.group, n1, n2, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n2 * [self.group.regular_repr]), 2),  # 56x56

            EquiResBlock(self.group, n2, n3, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n3 * [self.group.regular_repr]), 2),  # 28x28

            EquiResBlock(self.group, n3, n4, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n4 * [self.group.regular_repr]), 2),  # 14x14

            EquiResBlock(self.group, n4, n5, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n5 * [self.group.regular_repr]), 2),  # 7x7

            EquiResBlock(self.group, n5, n5, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n5 * [self.group.regular_repr]), 3),  # 2x2

            nn.R2Conv(
                nn.FieldType(self.group, n5 * [self.group.regular_repr]),
                nn.FieldType(self.group, n5 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n5 * [self.group.regular_repr]), inplace=True),
            # 2x2
            nn.PointwiseAvgPool(nn.FieldType(self.group, n5 * [self.group.regular_repr]), 2),  # → 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        if isinstance(x, torch.Tensor):
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 76x76
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 72x72
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 36x36
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 18x18
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 9x9
            EquiResBlock(self.group, n_out // 2, n_out, initialize=True),
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

class EquivariantVoxelEncoder58Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 58
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            # 56
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            # 28
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 14
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            # 12
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 6
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
    
class EquivariantVoxelEncoder64Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 64
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            # 32
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 16
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            # 6
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
    
class EquivariantVoxelEncoder16Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 16
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 6
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
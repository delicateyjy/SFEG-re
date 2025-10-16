"""
输出为sigmoid后的概率图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn import init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from thop import profile

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        # assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = KANLinear(num_channels, num_channels_reduced)
        self.fc2 = KANLinear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()

        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlpx = nn.Linear(patch_size, output_dim // 4)
        self.mlpy = nn.Linear(patch_size, output_dim // 4)
        self.mlp2 = nn.Linear(output_dim // 4, output_dim)
        self.mlp4 = nn.Linear(output_dim // 4, output_dim)
        self.norm2 = nn.LayerNorm(output_dim // 4)
        self.norm4 = nn.LayerNorm(output_dim // 4)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)


    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        # Local branch
        local_patches_x = x.unfold(2, P, P)
        local_patches_y = x.unfold(3, P, P)
        local_patches_x = local_patches_x.reshape(B, C, -1, P)
        local_patches_y = local_patches_y.reshape(B, C, -1, P)
        local_patches_x = local_patches_x.mean(dim=1)
        local_patches_y = local_patches_y.mean(dim=1)
        local_patches_x = self.mlpx(local_patches_x)
        local_patches_y = self.mlpy(local_patches_y)
        if P == 2:
            local_patches = torch.cat((local_patches_x, local_patches_y), dim=1)
            local_patches = self.norm2(local_patches)
            local_patches = self.mlp2(local_patches)
            local_attention = F.softmax(local_patches, dim=-1)
            local_out = local_patches * local_attention
            cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
            mask = cos_sim.clamp(0, 1)
            local_out = local_out * mask
            local_out = local_out @ self.top_down_transform
            local_out = local_out.reshape(B, H, W, self.output_dim)
            local_out = local_out.permute(0, 3, 1, 2)
            output = self.conv2(local_out)
            return output
        if P == 4:
            local_patches_y = TF.resize(local_patches_y, size=local_patches_x.shape[1:])
            local_patches = local_patches_x + local_patches_y
            local_patches = self.norm4(local_patches)
            local_patches = self.mlp4(local_patches)
            local_attention = F.softmax(local_patches, dim=-1)
            local_out = local_patches * local_attention
            cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
            mask = cos_sim.clamp(0, 1)
            local_out = local_out * mask
            local_out = local_out @ self.top_down_transform
            local_out = local_out.reshape(B, H // 2, W // 2, self.output_dim)
            local_out = local_out.permute(0, 3, 1, 2)
            local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
            output = self.conv4(local_out)
            return output


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class MDHA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)
        self.se = SELayer(filters, reduction_ratio=8)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.bn3 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.35)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.c2(x1)
        x2 = (x1 + x2 + x_skip) + (x_lga2 + x_lga4)
        x2 = self.drop(x2)
        x3 = self.c3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x = self.se(x3)

        return x

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.sigmod = nn.Sigmoid()
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.sigmod(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

class DSRB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1):
        super(DSRB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.agca = AGCA(in_channel=channels, ratio=4)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        tmp = x
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        res = 2 * self.sigmod(tmp - (x1 + x2 + x3 + x4) / 4) - 1
        x = self.relu(tmp * res)
        x = self.agca(x)
        return x

class CDSNET(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=[32, 64, 128, 256]):

        super(CDSNET, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList()
        self.sspcab = nn.ModuleList()
        for feature in reversed(features):
            self.sspcab.append(DSRB(channels=feature))

        # Down part
        for feature in features:
            self.downs.append(MDHA(in_channels, feature))
            in_channels = feature

        # UP part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(MDHA(feature * 2, feature))

        self.bottleneck = nn.Sequential(MDHA(features[-1], features[-1] * 2), DSRB(channels=features[-1]*2))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # UP part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            skip_connection = self.sspcab[idx // 2](skip_connection)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class exfloss(nn.Module):
    def __init__(self):
        super(exfloss, self).__init__()
    def forward(self, my_preds, gt_preds):
        assert my_preds.shape == gt_preds.shape
        eps = 0.00001
        my_preds = my_preds.clamp(min=eps, max=1.0 - eps)
        alpha = torch.tensor(gt_preds == 0).sum() / torch.tensor(gt_preds == 1).sum()
        alpha = torch.sqrt(alpha)
        beta = 1 + torch.sqrt(alpha)
        y1 = torch.ones_like(gt_preds)
        y0 = torch.zeros_like(gt_preds)
        threshold = 0.9
        w1 = torch.where(my_preds <= threshold, y1, y0)
        w0 = torch.where(my_preds > threshold, y1, y0)
        loss = -alpha * gt_preds * torch.log(my_preds)
        loss -= (1 - gt_preds) * w1 * torch.log(1 - my_preds)
        loss -= (1 - gt_preds) * w0 * beta * torch.log(1 - my_preds)
        loss = loss.mean()
        return loss
    
def test():
    x = torch.randn((1, 3, 512, 512)).cuda()  # batch size, channel,height,width

    model = CDSNET(in_channels=3, out_channels=1).cuda()
    flops, params = profile(model, (x,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    preds = model(x)
    print(x.shape)
    print(preds.shape)

# flops: 90.99 G, params: 14.73 M
# torch.Size([1, 3, 512, 512])
# torch.Size([1, 1, 512, 512])

if __name__ == "__main__":
    test()

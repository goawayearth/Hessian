import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c*2, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Hessian特征计算模块
        self._init_hessian_layers()
        
        """ 修改后的编码器 (第一个模块输入通道改为6) """
        self.e1 = encoder_block(6, 64)  # 原3通道 + Hessian3通道
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def _init_hessian_layers(self):
        """初始化Hessian计算层"""
        # 灰度转换
        self.gray_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        gray_weight = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
        self.gray_conv.weight.data = gray_weight
        self.gray_conv.weight.requires_grad = False

        # Hessian二阶导数核
        self._create_fixed_conv('xx', torch.tensor([[[0,0,0], [1,-2,1], [0,0,0]]]))
        self._create_fixed_conv('xy', torch.tensor([[[1,0,-1], [0,0,0], [-1,0,1]]])*0.25)
        self._create_fixed_conv('yy', torch.tensor([[[0,1,0], [0,-2,0], [0,1,0]]]))

    def _create_fixed_conv(self, name, kernel):
        """创建固定权重的卷积层"""
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        conv.weight.data = kernel.float().unsqueeze(0)
        conv.weight.requires_grad = False
        self.add_module(f'hessian_{name}', conv)

    def _compute_hessian(self, x):
        """计算Hessian特征"""
        gray = self.gray_conv(x)
        return torch.cat([
            self.hessian_xx(gray),
            self.hessian_xy(gray),
            self.hessian_yy(gray)
        ], dim=1)

    def forward(self, inputs):
        # 计算Hessian特征并与原图拼接
        hessian_feat = self._compute_hessian(inputs)
        fused_input = torch.cat([inputs, hessian_feat], dim=1)  # [B,6,H,W]

        """ Encoder """
        s1, p1 = self.e1(fused_input)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)  # torch.Size([2, 1, 512, 512])
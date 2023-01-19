import torch


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        # Each enc_conv/dec_conv block should look like this:
        # nn.Sequential(
        #     nn.Conv2d(...),
        #     ... (2 or 3 conv layers with relu and batchnorm),
        # )
        self.conv0 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)
        )

        # decoder (upsampling)
        self.upsample0 = nn.UpsamplingBilinear2d(size=(32, 32))  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.upsample1 = nn.UpsamplingBilinear2d(size=(64, 64)) # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.upsample2 = nn.UpsamplingBilinear2d(size=(128, 128))  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.upsample3 = nn.UpsamplingBilinear2d(size=(252, 252))   # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3),
            # nn.ReLU() no activation :(
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        # encoder
        e0 = self.pool0(self.conv0(x))
        e1 = self.pool1(self.enc_conv1(e0))
        e2 = self.pool2(self.enc_conv2(e1))
        e3 = self.pool3(self.enc_conv3(e2))

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b))
        d1 = self.dec_conv1(self.upsample1(d0))
        d2 = self.dec_conv2(self.upsample2(d1))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3
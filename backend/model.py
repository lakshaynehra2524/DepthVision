import torch 
import torch.nn as nn

# Creating encoder block : Which analyze the image and then resizes it to 1/2 shape (compress) by MaxPooling
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(),
            
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2))

    def forward(self, x):
        return self.block(x)
    

# Creating decoder block : Which analyze the image and then resizes it to double shape (expand) by ConvTranspose2d layer
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(

            # Upsample feature map
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
            nn.ReLU(),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU())

    def forward(self, x):
        return self.block(x)
    

# DepthVision Network 

class DepthVisionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = EncoderBlock(3, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),
        nn.ReLU(),
    
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.ReLU())

        # Decoder
        self.decoder1 = DecoderBlock(256,128)
        self.decoder2 = DecoderBlock(128,64)
        self.decoder3 = DecoderBlock(64,32)

        # Output layer
        self.output_layer = nn.Conv2d(32,1,kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        # Output
        x = self.output_layer(x)

        return x
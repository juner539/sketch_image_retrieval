import src.layers_unet as layers
import torch
from torch import nn
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.convLayer = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size, stride=stride, padding=padding)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.convLayer(x)
        x = F.relu(x)
        x = self.batchNorm(x)
        return x

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.convLayer1 = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size, stride=(1,1), padding=padding)
        self.batchNorm1 = nn.BatchNorm2d(in_channels)
        self.convLayer2 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size, stride=(1,1), padding=padding)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        res_x = x
        x = self.batchNorm1(x)
        x = F.leaky_relu(x)
        x = self.convLayer1(x)

        x = self.batchNorm2(x)
        x = F.leaky_relu(x)
        x = self.convLayer2(x)
        x = res_x + x
        
        return x

class resShortCutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.shortCutLayer = nn.Conv2d(in_channels, out_channels, 
                                    (1,1), stride=(2,2), padding=(0,0))
        self.convLayer1 = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size, stride=(2,2), padding=padding)
        self.batchNorm1 = nn.BatchNorm2d(in_channels)
        self.convLayer2 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size, stride=(1,1), padding=padding)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        res_x = self.shortCutLayer(x)
        
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.convLayer1(x)     
        
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.convLayer2(x)
        x = res_x + x

        return x

class resNet_256(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.convLayer = nn.Conv2d(1, 64, (3,3), stride=(2,1), padding=(1,1))
        self.batchNorm = nn.BatchNorm2d(64)
        self.resBlock1 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock2 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock3 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock4 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock5 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock6 = resBlock(64, 64, (3,3), padding=(1,1))
        self.resBlock7 = resShortCutBlock(64, 128, (3,3), padding=(1,1))
        self.resBlock8 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock9 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock10 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock11 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock12 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock13 = resBlock(128, 128, (3,3), padding=(1,1))
        self.resBlock14 = resShortCutBlock(128, 256, (3,3), padding=(1,1))
        self.resBlock15 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock16 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock17 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock18 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock19 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock20 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock21 = resBlock(256, 256, (3,3), padding=(1,1))
        self.resBlock22 = resBlock(256, 256, (3,3), padding=(1,1))
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,x):
        x = self.convLayer(x)
        x = F.relu(x)
        x = self.batchNorm(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.resBlock5(x)
        x = self.resBlock6(x)
        x = self.resBlock7(x)
        x = self.resBlock8(x)
        x = self.resBlock9(x)
        x = self.resBlock10(x)
        x = self.resBlock11(x)
        x = self.resBlock12(x)
        x = self.resBlock13(x)
        x = self.resBlock14(x)
        x = self.resBlock15(x)
        x = self.resBlock16(x)
        x = self.resBlock17(x)
        x = self.resBlock18(x)
        x = self.resBlock19(x)
        x = self.resBlock20(x)
        x = self.resBlock21(x)
        x = self.resBlock22(x)
        x = self.globalAvgPool(x)
        return x

class InitGenConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = layers._equalized_linear(in_channels, in_channels)
        self.conv1 = layers._equalized_deconv2d(in_channels, out_channels, (8, 25))
        self.pixNorm = layers.PixelwiseNorm()
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.pixNorm(x)
        x = self.fc(x.squeeze())
        x = self.lrelu(x)
        x = self.pixNorm(x)
        x = self.conv1(x.view(-1, x.shape[1], 1 , 1))
        x = self.lrelu(x)
        x = self.pixNorm(x)
        return x
    
class UpsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.conv1 = layers._equalized_conv2d(in_channels, out_channels, (3, 3), padding=(1,1))
        self.conv2 = layers._equalized_conv2d(out_channels, out_channels, (3, 3), padding=(1,1))
        self.pixNorm = layers.PixelwiseNorm()
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.upsampling(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.pixNorm(x)
        
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.pixNorm(x)
        return x

class charEncoder(nn.Module):
    def __init__(self, corpus):
        super().__init__()
        self.embed = nn.Embedding(len(corpus['char2idx']), 256)
        self.conv1 = nn.Conv1d(9, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.globalMaxPool = nn.AdaptiveMaxPool1d(1)
               
    def forward(self, x):
        x = self.embed(x)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.globalMaxPool(x)
        return x.unsqueeze(2)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.initBlock = InitGenConvBlock(512, 256)
        self.block1 = UpsamplingConvBlock(256, 256, 2)
        self.block2 = UpsamplingConvBlock(256, 256, 1)
        self.block2 = UpsamplingConvBlock(256, 256, 1)
        self.block3 = UpsamplingConvBlock(256, 128, 1)
        self.block4 = UpsamplingConvBlock(128, 128, 1)
        self.block5 = UpsamplingConvBlock(128, 128, 1)
        self.block6 = UpsamplingConvBlock(128, 128, 1)
        self.block7 = UpsamplingConvBlock(128, 64, 1)
        self.block8 = UpsamplingConvBlock(64, 64, 2)
        self.block9 = UpsamplingConvBlock(64, 64, 1)
        self.block10 = UpsamplingConvBlock(64, 64, 1)
        self.block11 = UpsamplingConvBlock(64, 32, 1)
        self.block12 = UpsamplingConvBlock(32, 32, 1)
        self.block13 = UpsamplingConvBlock(32, 32, 1)
        self.block14 = UpsamplingConvBlock(32, 32, 1)
        self.block15 = UpsamplingConvBlock(32, 16, 1)
        self.block16 = UpsamplingConvBlock(16, 16, 1)
        
        self.blockOut = layers._equalized_conv2d(16, 1, (1,1), bias=True)
        
    def forward(self, x):
        x = self.initBlock(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.blockOut(x)
        x = torch.tanh(x)
        
        return x
    
class TripleDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.initBlock = InitGenConvBlock(512, 256)
        self.block1 = UpsamplingConvBlock(256, 256, 2)
        self.block2 = UpsamplingConvBlock(256, 128, 1)
        self.block3 = UpsamplingConvBlock(128, 128, 1)
        self.block4 = UpsamplingConvBlock(128, 64, 1)
        self.block5 = UpsamplingConvBlock(64, 64, 2)
        self.block6 = UpsamplingConvBlock(64, 32, 1)
        self.block7 = UpsamplingConvBlock(32, 32, 1)
        self.block8 = UpsamplingConvBlock(32, 16, 1)
        
        self.initOut = layers._equalized_conv2d(256, 1, (1,1), bias=True)
        self.block4Out = layers._equalized_conv2d(128, 1, (1,1), bias=True)
        self.block8Out = layers._equalized_conv2d(16, 1, (1,1), bias=True)
        
    def forward(self, x):
        x = self.initBlock(x)
        initOut = self.initOut(x)
        initOut = torch.tanh(initOut)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        block4Out = self.block4Out(x)
        block4Out = torch.tanh(block4Out)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        block8Out = self.block8Out(x)
        block8Out = torch.tanh(block8Out)
        
        return (initOut, block4Out, block8Out)

class Generator(nn.Module):
    def __init__(self, corpus):
        super().__init__()
        self.imgEncoder = resNet_256().cuda()
        self.charEncoder = charEncoder(corpus).cuda()
        self.decoder = Decoder().cuda()
        
    def forward(self, x, txt):
        imgEmbed = self.imgEncoder(x)
        charEmbed = self.charEncoder(txt)
        embedding = torch.cat((imgEmbed, charEmbed), dim=1)
        outImg = self.decoder(embedding)
        return outImg

class InitDisConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.downsampling = nn.MaxPool2d(scale_factor)
        self.conv1 = layers._equalized_deconv2d(1, in_channels, (1, 1))
        self.conv2 = layers._equalized_conv2d(in_channels, in_channels, (3, 3), padding=(1,1))
        self.conv3 = layers._equalized_conv2d(in_channels, out_channels, (3, 3), padding=(1,1))
        
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.downsampling(x)
        return x

class InitDisConvMarkovBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.downsampling = nn.MaxPool2d(scale_factor)
        self.conv1 = layers._equalized_deconv2d(1, in_channels, (1, 1))
        self.conv2 = layers._equalized_conv2d(in_channels, in_channels, (3, 3), padding=(1,1))
        self.conv3 = layers._equalized_conv2d(in_channels, out_channels, (1, 1))
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.downsampling(x)
        return x

class DownsamplingConvMarkovBlock(nn.Module):     
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.downsampling = nn.MaxPool2d(scale_factor)
        self.conv0 = layers._equalized_conv2d(in_channels, in_channels, (1, 1))   
        self.conv1 = layers._equalized_conv2d(in_channels, in_channels, (3, 3), padding=(1,1))
        self.conv2 = layers._equalized_conv2d(in_channels, out_channels, (1, 1))

        self.lrelu = nn.LeakyReLU(0.2) 
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.downsampling(x)
        return x    
    
class DownsamplingConvBlock(nn.Module):     
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.downsampling = nn.MaxPool2d(scale_factor)
        self.conv1 = layers._equalized_conv2d(in_channels, in_channels, (3, 3), padding=(1,1))
        self.conv2 = layers._equalized_conv2d(in_channels, out_channels, (3, 3), padding=(1,1))

        self.lrelu = nn.LeakyReLU(0.2) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.downsampling(x)
        return x

class FinalDisConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mnBtchStdDv = layers.MinibatchStdDev()
        self.conv1 = layers._equalized_conv2d(in_channels+1, in_channels, (3, 3), padding=(1,1))
        self.conv2 = layers._equalized_conv2d(in_channels, in_channels, (8, 25))
        self.fc = layers._equalized_linear(in_channels, 1)
        
        self.lrelu = nn.LeakyReLU(0.2) 
        
    def forward(self, x):
        x = self.mnBtchStdDv(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
class FinalDisConvMarkovBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mnBtchStdDv = layers.MinibatchStdDev()
        self.conv0 = layers._equalized_conv2d(in_channels+1, in_channels, (1, 1))   
        self.conv1 = layers._equalized_conv2d(in_channels, in_channels, (3, 3), padding=(1,1))
        self.conv2 = layers._equalized_conv2d(in_channels, 1, (1, 1))
        self.lrelu = nn.LeakyReLU(0.2) 
        
    def forward(self, x):
        x = self.mnBtchStdDv(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.initBlock = InitDisConvBlock(16, 32, 1)
        self.block1 = DownsamplingConvBlock(32, 32, 2)
        self.block2 = DownsamplingConvBlock(32, 64, 1)
        self.block3 = DownsamplingConvBlock(64, 64, 1)
        self.block4 = DownsamplingConvBlock(64, 128, 1)
        self.block5 = DownsamplingConvBlock(128, 128, 2)
        self.block6 = DownsamplingConvBlock(128, 256, 1)
        self.block7 = DownsamplingConvBlock(256, 256, 1)
        self.block8 = DownsamplingConvBlock(256, 512, 1)
        self.finalBlock = FinalDisConvBlock(512)
        
    def forward(self, x):
        x = self.initBlock(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.finalBlock(x)
        return x

class MarkovConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        return x
    
class MarkovDiscriminator(nn.Module):
    def __init__(self, first_channels):
        super().__init__()
        self.block1 = MarkovConvBlock(3, first_channels, 2)
        self.block2 = MarkovConvBlock(first_channels, first_channels*2, 2) 
        self.block3 = MarkovConvBlock(first_channels*2, first_channels*4, (2,1)) 
        self.block4 = MarkovConvBlock(first_channels*4, first_channels*8, (2,1)) 
        self.conv5 = nn.Conv2d(in_channels=first_channels*8, out_channels=1, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        return x
    
class SRNet_BG(nn.Module):
    def __init__(self, res_layers, first_channels):
        super().__init__()
        self.downBlock1 = DownsamplingConvBlock(3, first_channels, 2)
        self.downBlock2 = DownsamplingConvBlock(first_channels, first_channels*2, 2)
        self.downBlock3 = DownsamplingConvBlock(first_channels*2, first_channels*4, 2)
        
        self.resBlocks = nn.ModuleList([])
        for layer in range(res_layers):
            self.resBlocks.append(resBlock(first_channels*4, first_channels*4, (3,3), padding=(1,1)))
        
        self.upBlock1 = UpsamplingConvBlock(first_channels*8, first_channels*2, 2)
        self.upBlock2 = UpsamplingConvBlock(first_channels*4, first_channels, 2)
        self.upBlock3 = UpsamplingConvBlock(first_channels*2, first_channels*2, 2)
        self.conv4 = nn.Conv2d(first_channels*2, 1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.tanH = nn.Tanh()
        
    def forward(self, x):
        d1 = self.downBlock1(x)
        d2 = self.downBlock2(d1)
        d3 = self.downBlock3(d2)
        x = d3

        for block in self.resBlocks:
            x = block(x)
        
        u1 = self.upBlock1(torch.cat((x, d3), dim=1))
        u2 = self.upBlock2(torch.cat((u1, d2), dim=1))
        u3 = self.upBlock3(torch.cat((u2, d1), dim=1))

        x = self.conv4(u3)
        x = self.bn4(x)
        x = self.tanH(x)
        return x

from torchsummary import summary
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = SRNet_BG(5, 16).to(device)
    print(summary(G, input_size=(3,224,224)))
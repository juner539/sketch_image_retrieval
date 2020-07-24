import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class VggNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VggNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(4096, 2048))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class UpsampleConvBlock(nn.Module):
    def __init__(self, up_size=0, in_channel=0, out_channel=0, kernel_size=1, stride=1, tanh=False):
        super(UpsampleConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size, up_size)),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),

        )
        if tanh:
            self.block = nn.Sequential(
                nn.UpsamplingNearest2d(size=(up_size, up_size)),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=stride, bias=False),
                nn.Tanh(),
            )
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.vggnetfeats = VggNetFeats()
        self.upsampleConvBlock1 = UpsampleConvBlock(up_size=1, in_channel=2048,
                                                    out_channel=1024, kernel_size=1, stride=1)
        self.upsampleConvBlock2 = UpsampleConvBlock(up_size=7, in_channel=1024,
                                                    out_channel=512, kernel_size=1, stride=1)
        self.upsampleConvBlock3 = UpsampleConvBlock(up_size=14, in_channel=512,
                                                    out_channel=256, kernel_size=1, stride=1)
        self.upsampleConvBlock4 = UpsampleConvBlock(up_size=28, in_channel=256,
                                                    out_channel=128, kernel_size=3, stride=2)
        self.upsampleConvBlock5 = UpsampleConvBlock(up_size=56, in_channel=128,
                                                    out_channel=64, kernel_size=3, stride=2)
        self.upsampleConvBlock6 = UpsampleConvBlock(up_size=112, in_channel=64,
                                                    out_channel=32, kernel_size=3, stride=2)
        self.upsampleConvBlock7 = UpsampleConvBlock(up_size=224, in_channel=32,
                                                    out_channel=1, kernel_size=1, stride=1, tanh=True)
    def forward(self, x):
        x = self.vggnetfeats(x)
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.upsampleConvBlock1(x)
        x = self.upsampleConvBlock2(x)
        x = self.upsampleConvBlock3(x)
        x = self.upsampleConvBlock4(x)
        x = self.upsampleConvBlock5(x)
        x = self.upsampleConvBlock6(x)
        x = self.upsampleConvBlock7(x)
        return x



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    print(G)
    print(summary(G, input_size=(3, 224, 224)))
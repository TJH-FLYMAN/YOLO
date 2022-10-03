import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import *

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads=heads
        self.query=nn.Conv2d(n_dims,n_dims,kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        #torch.randn:用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1）。
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        #TENSOR的乘法。torch.matmul
        #将tensor的维度换位。permute函数
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        #view是填充  permute是转换
        #所以，一般需要将tensor拉开时使用view()，而在需要转置时使用permute()
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

#1*1 3*3 bn shortcut
class Bottleneck(nn.Module):
    expansion = 1  #每个stage中纬度拓展的倍数

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes[1], kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DarkNet(nn.Module):
    def __init__(self,layers,resolution=(224, 224), heads=4):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.resolution=list(resolution)
        # 416,416,3 -> 416,416,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0],stride=1)
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1],stride=2)
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2],stride=2)
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3],stride=2)
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024],layers[4],stride=2,heads=heads,mhsa=True)

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,planes,num_blocks,stride=1):
        layers=[]
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
        #self.inplanes = planes * block.expansion
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes=planes[1]

        for i in range(0, num_blocks):    #blocks就是残差块堆叠的次数
            layers.append(("residual_{}".format(i), Bottleneck(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):

        x = self.maxpool(self.relu1(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

# def darknet53(pretrained,resolution=(224, 224), heads=4):
#     model=DarkNet(Bottleneck,[1,2,8,8,4],resolution=resolution,heads=heads)
#     if pretrained:
#         if isinstance(pretrained, str):
#             model.load_state_dict(torch.load(pretrained))
#         else:
#             raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
#     return model

def darknet53(resolution=(224, 224), heads=4):
    return DarkNet([1,2,8,8,4],resolution=resolution,heads=heads)

def main():
    x = torch.randn([2, 3, 224, 224])
    model = darknet53()
    print(model(x))
    print(get_n_params(model))

if __name__ == '__main__':
    main()

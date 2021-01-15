import torch
import torch.nn as nn
from torchvision.models import vgg19

class DCTGAN_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.relu_1 = nn.PReLU(init=0.2)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU(0.2)

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.relu_3 = nn.PReLU(init=0.2)
        
        self.conv_res = StraightBlock(kernel_size=3, block_size=11)

        self.convt_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt_2 = nn.BatchNorm2d(128)
        self.relut_2 = nn.PReLU(init=0.2)

        self.convt_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt_1 = nn.BatchNorm2d(64)
        self.relut_1 = nn.PReLU(init=0.2)
        
        self.conv_output = nn.Conv2d(64, 3, kernel_size=7, padding=3)
        self.out = nn.Tanh()

    def forward(self, img_input):
        conv_1_out = self.relu_1(self.conv_1(img_input))
        conv_2_out = self.relu_2(self.bn_2(self.conv_2(conv_1_out)))
        conv_3_out = self.relu_3(self.bn_3(self.conv_3(conv_2_out)))
        conv_res_out = self.conv_res(conv_3_out)
        convt_2_out = self.relut_2(self.bnt_2(self.convt_2(conv_res_out)))
        convt_1_out = self.relut_1(self.bnt_1(self.convt_1(convt_2_out)))
        output = self.out(self.conv_output(convt_1_out)) + img_input

        return output



class DCTGAN_D1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 256,256 --> 128, 128
        self.relu_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 128, 128 --> 64, 64
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.LeakyReLU(0.2)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.LeakyReLU(0.2)
        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64, 64 --> 32, 32
        self.bn_5 = nn.BatchNorm2d(128)
        self.relu_5 = nn.LeakyReLU(0.2)
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(128)
        self.relu_6 = nn.LeakyReLU(0.2)
        self.dense1 = nn.Linear(128 * 32 * 32, 1024)
        self.relu_7 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.out = nn.Sigmoid()

    def forward(self, img_input):
        conv_1_out = self.relu_1(self.conv_1(img_input))
        conv_2_out = self.relu_2(self.bn_2(self.conv_2(conv_1_out)))
        conv_3_out = self.relu_3(self.bn_3(self.conv_3(conv_2_out)))
        conv_4_out = self.relu_4(self.bn_4(self.conv_4(conv_3_out)))
        conv_5_out = self.relu_5(self.bn_5(self.conv_5(conv_4_out)))
        conv_6_out = self.relu_6(self.bn_6(self.conv_6(conv_5_out)))
        #self.in_shape = c * h * w
        conv_6_out_reshape = conv_6_out.reshape(-1, 128 * 32 * 32)
        dense1_out = self.relu_7(self.dense1(conv_6_out_reshape))
        output = self.out(self.dense2(dense1_out))

        return output


class DCTGAN_D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 256,256 --> 128, 128
        self.relu_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 128, 128 --> 64, 64
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.LeakyReLU(0.2)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.LeakyReLU(0.2)
        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64, 64 --> 32, 32
        self.bn_5 = nn.BatchNorm2d(128)
        self.relu_5 = nn.LeakyReLU(0.2)
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(128)
        self.relu_6 = nn.LeakyReLU(0.2)
        self.dense1 = nn.Linear(128 * 32 * 32, 1024)
        self.relu_7 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.out = nn.Sigmoid()

    def forward(self, img_input):
        conv_1_out = self.relu_1(self.conv_1(img_input))
        conv_2_out = self.relu_2(self.bn_2(self.conv_2(conv_1_out)))
        conv_3_out = self.relu_3(self.bn_3(self.conv_3(conv_2_out)))
        conv_4_out = self.relu_4(self.bn_4(self.conv_4(conv_3_out)))
        conv_5_out = self.relu_5(self.bn_5(self.conv_5(conv_4_out)))
        conv_6_out = self.relu_6(self.bn_6(self.conv_6(conv_5_out)))
        
        #self.in_shape = c * h * w
        conv_6_out_reshape = conv_6_out.reshape(-1, 128 * 32 * 32)
        dense1_out = self.relu_7(self.dense1(conv_6_out_reshape))
        output = self.out(self.dense2(dense1_out))

        return output


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class StraightBlock(nn.Module):
    def __init__(self, kernel_size, block_size):
        super().__init__()
        layers = []
        for i in range(block_size):
            layers += [ConvBlock(kernel_size)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, img_input):
        return self.blocks(img_input)

class ConvBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(256),
            nn.PReLU(init=0.2),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(256)
        )
        self.last_relu = nn.PReLU(init=0.2)
        

    def forward(self, img_input):
        blocks_out = self.blocks(img_input)
        return self.last_relu(blocks_out + img_input)



if __name__ == "__main__":
    img_input = torch.rand([4, 3, 256, 256])
    dis = Generator()
    img_out = dis(img_input)

    print(img_out.shape)

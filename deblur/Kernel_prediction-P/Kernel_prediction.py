import torch
import torch.nn as nn

class Kernel_prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)#, padding=3)
        self.relu_1 = nn.ReLU(96)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)#, padding=2)
        self.relu_2 = nn.ReLU(256)
        
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dense1 = nn.Linear(256 * 5 * 5, 1024)
        
        self.dense2 = nn.Linear(1024, 1)

    def forward(self, img_input):
        layer_1_out = self.relu_1(self.conv_1(img_input))
        layer_2_out = self.pool1(layer_1_out)
        layer_3_out = self.relu_2(self.conv_2(layer_2_out))
        layer_4_out = self.pool2(layer_3_out)
        layer_4_out_reshape = layer_4_out.reshape(-1, 256 * 5 * 5)
        layer_5_out = self.dense1(layer_4_out_reshape)
        output = self.dense2(layer_5_out)

        return output

if __name__ == "__main__":
    img_input = torch.rand([4, 3, 256, 256])
    dis = Generator()
    img_out = dis(img_input)

    print(img_out.shape)

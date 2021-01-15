from torch.autograd import Variable

def rgb_to_ycbcr(img_input):
    output = Variable(img_input.data.new(*img_input.size()))
    output[:, 0, :, :] = img_input[:, 0, :, :] * (65.481 / 255) + img_input[:, 1, :, :] * (128.553 / 255) + img_input[:, 2, :, :] * (24.966 / 255) + (16 / 255)

    return output
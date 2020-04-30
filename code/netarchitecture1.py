import torch.nn as nn


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]
    return h, w


def weights_init(neural_net):
    """ Initializes the weights of the neural network
    :param neural_net: (De-)Convolutional Neural Network where weights should be initialized
    """
    classname = neural_net.__class__.__name__
    if classname.find('Conv') != -1:
        neural_net.weight.data.normal_(0, 2e-2)
    elif classname.find('BatchNorm') != -1:
        neural_net.weight.data.normal_(1, 2e-2)
        neural_net.bias.data.fill_(0)


class PrintLayer(nn.Module):
    """ helper to access params during training """
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        # Do your print / debug stuff here
        print("Print Layer out in forward step", str(x.shape))
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ### torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            nn.ConvTranspose2d(100, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            PrintLayer(),
            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Upsample((800,600)),
            PrintLayer(),
            nn.Tanh()
        )
    def forward(self, input_vector):
        return self.main(input_vector)

#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 8, 4, 1, bias=False),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(64, 128, 6, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(128, 256, 8, 4, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            PrintLayer()
            #nn.Sigmoid()
        )
        self.fully = nn.Sequential(
            nn.Linear(56,2),
            #PrintLayer(),
            nn.Sigmoid())


    def forward(self, input_image):
        #print("in forward neural net: "+str(self.main(input_image).shape))
        return self.fully(self.main(input_image).view(-1))

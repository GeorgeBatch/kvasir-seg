import torch
import torchsummary

# --------------------------------------------------------------------------------
# Basic UNet architecture
# --------------------------------------------------------------------------------

# The output channel for segmentation should be equal to number of classes we want to segment the image into.
#   For a binary segmentation, this is a value of 1.
class UNet(torch.nn.Module):

    def conv_block(self, channel_in, channel_out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(channel_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(channel_out),
            torch.nn.ReLU(inplace=True)
        )


    def __init__(self, channel_in, channel_out, bilinear=None):
        super(UNet, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        
        # initial convolutional block
        self.initial = self.conv_block(channel_in, 64)
        
        # encoder layers
        self.down0 = self.conv_block(64, 128)
        self.down1 = self.conv_block(128, 256)
        self.down2 = self.conv_block(256, 512)
        self.down3 = self.conv_block(512, 1024)
        
        # decoder layers
        self.up0_0 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up0_1 = self.conv_block(1024, 512)
        self.up1_0 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_1 = self.conv_block(512, 256)
        self.up2_0 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2_1 = self.conv_block(256, 128)
        self.up3_0 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3_1 = self.conv_block(128, 64)
        
        # final layer before output
        self.final = torch.nn.Conv2d(64, channel_out, kernel_size=1)

    def forward(self,x):
        "Forward pass"
        x_in= self.initial(x)
        enc0 = self.down0(torch.nn.MaxPool2d(2)(x_in))
        enc1 = self.down1(torch.nn.MaxPool2d(2)(enc0))
        enc2 = self.down2(torch.nn.MaxPool2d(2)(enc1))
        enc3 = self.down3(torch.nn.MaxPool2d(2)(enc2))
        
        dec0 = self.up0_0(enc3)
        diff0 = torch.FloatTensor(list(enc2.size())[2:]) - torch.FloatTensor(list(dec0.shape))[2:]
        dec0 = torch.nn.functional.pad(dec0, (int((diff0/2).floor()[0]), int((diff0/2).ceil()[0]), int((diff0/2).floor()[1]), int((diff0/2).ceil()[1])))
        dec0 = self.up0_1(torch.cat((enc2, dec0), dim=1))

        dec1 = self.up1_0(dec0)
        diff1 = torch.FloatTensor(list(enc1.size())[2:]) - torch.FloatTensor(list(dec1.shape))[2:]
        dec1 = torch.nn.functional.pad(dec1, (int((diff1/2).floor()[0]), int((diff1/2).ceil()[0]), int((diff1/2).floor()[1]), int((diff1/2).ceil()[1])))
        dec1 = self.up1_1(torch.cat((enc1, dec1), dim=1))

        dec2 = self.up2_0(dec1)
        diff2 = torch.FloatTensor(list(enc0.size())[2:]) - torch.FloatTensor(list(dec2.shape))[2:]
        dec2 = torch.nn.functional.pad(dec2, (int((diff2/2).floor()[0]), int((diff2/2).ceil()[0]), int((diff2/2).floor()[1]), int((diff2/2).ceil()[1])))
        dec2 = self.up2_1(torch.cat((enc0, dec2), dim=1))

        dec3 = self.up3_0(dec2)
        diff3 = torch.FloatTensor(list(x.size())[2:]) - torch.FloatTensor(list(dec3.shape))[2:]
        dec3 = torch.nn.functional.pad(dec3, (int((diff3/2).floor()[0]), int((diff3/2).ceil()[0]), int((diff3/2).floor()[1]), int((diff3/2).ceil()[1])))
        dec3 = self.up3_1(torch.cat((x_in, dec3), dim=1))
        
        x_out = self.final(dec3) # ? no activation here
        return x_out


# --------------------------------------------------------------------------------
# UNet with an attempt at an attention mechanism
# --------------------------------------------------------------------------------

def compute_attention_mask(dec):
    # needed in forward pass of UNet_attention network
    original_dtype = dec.dtype
    dec = torch.sigmoid(dec)             # squash into [0, 1]
    dec = (dec > 0.5).to(original_dtype) # binarize and return to original type
    return dec

class UNet_attention(UNet):
    def __init__(self, channel_in, channel_out, bilinear=None):
        # everything stays the same except for the forward pass
        UNet.__init__(self, channel_in, channel_out)


    def forward(self,x):
        "Forward pass"
        x_in= self.initial(x)
        enc0 = self.down0(torch.nn.MaxPool2d(2)(x_in))
        enc1 = self.down1(torch.nn.MaxPool2d(2)(enc0))
        enc2 = self.down2(torch.nn.MaxPool2d(2)(enc1))
        enc3 = self.down3(torch.nn.MaxPool2d(2)(enc2))
        
        dec0 = self.up0_0(enc3)
        dec0 = compute_attention_mask(dec0)
        diff0 = torch.FloatTensor(list(enc2.size())[2:]) - torch.FloatTensor(list(dec0.shape))[2:]
        dec0 = torch.nn.functional.pad(dec0, (int((diff0/2).floor()[0]), int((diff0/2).ceil()[0]), int((diff0/2).floor()[1]), int((diff0/2).ceil()[1])))
        dec0 = self.up0_1(torch.cat((enc2, enc2*dec0), dim=1))

        dec1 = self.up1_0(dec0)
        dec1 = compute_attention_mask(dec1)
        diff1 = torch.FloatTensor(list(enc1.size())[2:]) - torch.FloatTensor(list(dec1.shape))[2:]
        dec1 = torch.nn.functional.pad(dec1, (int((diff1/2).floor()[0]), int((diff1/2).ceil()[0]), int((diff1/2).floor()[1]), int((diff1/2).ceil()[1])))
        dec1 = self.up1_1(torch.cat((enc1, enc1*dec1), dim=1))

        dec2 = self.up2_0(dec1)
        dec2 = compute_attention_mask(dec2)
        diff2 = torch.FloatTensor(list(enc0.size())[2:]) - torch.FloatTensor(list(dec2.shape))[2:]
        dec2 = torch.nn.functional.pad(dec2, (int((diff2/2).floor()[0]), int((diff2/2).ceil()[0]), int((diff2/2).floor()[1]), int((diff2/2).ceil()[1])))
        dec2 = self.up2_1(torch.cat((enc0, enc0*dec2), dim=1))

        dec3 = self.up3_0(dec2)
        dec3 = compute_attention_mask(dec3)
        diff3 = torch.FloatTensor(list(x.size())[2:]) - torch.FloatTensor(list(dec3.shape))[2:]
        dec3 = torch.nn.functional.pad(dec3, (int((diff3/2).floor()[0]), int((diff3/2).ceil()[0]), int((diff3/2).floor()[1]), int((diff3/2).ceil()[1])))
        dec3 = self.up3_1(torch.cat((x_in, x_in*dec3), dim=1))
        
        x_out = self.final(dec3) # no activation here
        return x_out

    
if __name__ == "__main__":   
    # --------------------------------------------------------------------------------
    # Check for UNet
    # --------------------------------------------------------------------------------

    model = UNet(channel_in=3, channel_out=1)
    torchsummary.summary(model, (3, 256, 256), device='cpu') # Runs a basic test of the network. Very useful and clear!
    
    # --------------------------------------------------------------------------------
    # Check for UNet with an attempt at attention
    # --------------------------------------------------------------------------------

    model = UNet_attention(channel_in=3, channel_out=1)
    torchsummary.summary(model, (3, 256, 256), device='cpu') # Runs a basic test of the network. Very useful and clear!
    
    print("\nExample of attention mask:")
    
    # initialize BATCH * Channel * H * W
    enc2 = torch.FloatTensor(1, 1, 3, 6).uniform_(-10, 10)
    print("enc2\n", enc2)
    dec0 = torch.FloatTensor(1, 1, 2, 5).uniform_(-10, 10)
    print("dec0\n", dec0)

    print("dec0.dtype:", dec0.dtype)

    # compute padding margins
    diff0 = torch.FloatTensor(list(enc2.size())[2:]) - torch.FloatTensor(list(dec0.shape))[2:]

    # # squash into [0, 1]
    # dec0 = torch.sigmoid(dec0)
    # # binarize
    # dec0 = (dec0 > 0.5).to(torch.float)
    # # pad with zeros
    dec0 = compute_attention_mask(dec0)

    dec0 = torch.nn.functional.pad(dec0, (int((diff0/2).floor()[0]), int((diff0/2).ceil()[0]), int((diff0/2).floor()[1]), int((diff0/2).ceil()[1])))
    print("dec0 after attention and padding\n", dec0)

    # attention on high-res features
    print("attention on high-res features\n", enc2*dec0)

    # what goes as input further
    print("what goes as input further\n", torch.cat((enc2, enc2*dec0), dim=1))
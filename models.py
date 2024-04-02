import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorAtL2H(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=15):
        super(GeneratorAtL2H, self).__init__()
        in_features = 64
        # Initial convolution block       
        model_1 = [ nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        for _ in range(n_residual_blocks):
            model_1 += [ResidualBlock(in_features)]

        self.model_1 = nn.Sequential(*model_1) 
        # Upsampling
        model_2 = []
        out_features = in_features
        for _ in range(2):
            model_2 += [  nn.Upsample(scale_factor=2.0, mode='bicubic'),
                        nn.Conv2d(in_features, out_features, 7, stride=1, padding=3),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
        
        # Output layer  
        self.model_2 = nn.Sequential(*model_2, 
                        nn.ReflectionPad2d(3),                    
                        nn.Conv2d(in_features, 7, 7),
                        nn.Tanh())           
        
        self.attention = nn.Sequential(*model_2, 
                            nn.ReflectionPad2d(3), 
                            nn.Conv2d(in_features, 8, 7),
                            nn.Softmax(dim=1))  
 
    def forward(self, x):
        image = self.model_2(self.model_1(x))
        attention = self.attention(self.model_1(x))

        image1 = image[:, 0:1, :, :]
        image2 = image[:, 1:2, :, :]
        image3 = image[:, 2:3, :, :]
        image4 = image[:, 3:4, :, :]
        image5 = image[:, 4:5, :, :]
        image6 = image[:, 5:6, :, :]
        image7 = image[:, 6:7, :, :]
        # image8 = image[:, 7:8, :, :]

        attention1 = attention[:, 0:1, :, :]
        attention2 = attention[:, 1:2, :, :]
        attention3 = attention[:, 2:3, :, :]
        attention4 = attention[:, 3:4, :, :]
        attention5 = attention[:, 4:5, :, :]
        attention6 = attention[:, 5:6, :, :]
        attention7 = attention[:, 6:7, :, :]
        attention8 = attention[:, 7:8, :, :]

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        # output8 = image8 * attention8
        output8 = nn.functional.interpolate(input=x,scale_factor=4,mode='bicubic') * attention8

        o=output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 
               
        return o, output1, output2, output3, output4, output5, output6, output7, output8, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8, image1, image2,image3,image4,image5,image6,image7



class GeneratorAtH2L(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=15):
        super(GeneratorAtH2L, self).__init__()
        in_features = 64
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        out_features = in_features
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 7, stride=2, padding=3),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        self.model_1 = nn.Sequential(*model)

        # Output layer       
        self.model_2 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, 7, 3),
                    nn.Tanh() )

        self.attention = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, 8, 3),
                    nn.Softmax(dim=1) )

    def forward(self, x):
        image = self.model_2(self.model_1(x))
        attention = self.attention(self.model_1(x))

        image1 = image[:, 0:1, :, :]
        image2 = image[:, 1:2, :, :]
        image3 = image[:, 2:3, :, :]
        image4 = image[:, 3:4, :, :]
        image5 = image[:, 4:5, :, :]
        image6 = image[:, 5:6, :, :]
        image7 = image[:, 6:7, :, :]
        # image8 = image[:, 7:8, :, :] 

        attention1 = attention[:, 0:1, :, :]
        attention2 = attention[:, 1:2, :, :]
        attention3 = attention[:, 2:3, :, :]
        attention4 = attention[:, 3:4, :, :]
        attention5 = attention[:, 4:5, :, :]
        attention6 = attention[:, 5:6, :, :]
        attention7 = attention[:, 6:7, :, :]
        attention8 = attention[:, 7:8, :, :]

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        # output8 = image8 * attention8
        output8 = nn.functional.interpolate(input=x,scale_factor=0.25,mode='bicubic') * attention8
   
        o=output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 
               
        return o, output1, output2, output3, output4, output5, output6, output7, output8, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8, image1, image2,image3,image4,image5,image6,image7




class DiscriminatorH(nn.Module):
    def __init__(self, input_nc):
        super(DiscriminatorH, self).__init__()
        in_features = 64
        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, in_features, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features, in_features*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*2, in_features*4, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*4, in_features*8, 4, padding=1),
                    nn.InstanceNorm2d(in_features*8), 
                    nn.LeakyReLU(0.2, inplace=True) ]


        model += [nn.Conv2d(in_features*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class DiscriminatorL(nn.Module):
    def __init__(self, input_nc):
        super(DiscriminatorL, self).__init__()
        in_features = 64
        # A bunch of convolutions one after another

        model = [  nn.Conv2d(input_nc, in_features, 3, stride=2, padding=1), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features, in_features*2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*2, in_features*4, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*4, in_features*8, 3, padding=1),
                    nn.InstanceNorm2d(in_features*8), 
                    nn.LeakyReLU(0.2, inplace=True) ]


        model += [nn.Conv2d(in_features*8, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    


# from torchsummary import summary

# summary(GeneratorAtL2H(1,1).to('cuda'),input_size=(1,64,64))
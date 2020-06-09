import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageEncoder(nn.Module):
    """
    This class will Encode the Image and produce Feature vectors
    """

    def __init__(self, feature_map_size=14):
        """
        Input (Optional):
        -----------------
        feature_map_size : scaler value  
        """
        super(ImageEncoder, self).__init__()
        self.enc_image_size = feature_map_size

        #use pretrained resnet 101 
        resnet101 = torchvision.models.resnet101(pretrained=True)
        
        #removed last 2 classfication layers
        features_extracter = list(resnet101.children())[:-2]

        self.resnet = nn.Sequential(*features_extracter)

        #in paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" has suggested to use 14x14 feature_map
        #paper has used 512 channel for feature map, but using resnet101 , i am using 2048 channel.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_map_size, feature_map_size))

        self.finetune()

    def forward(self, images):
        """
        Input:
        -----
        Image Input Dimension: [N,RGB,W,H]
        where N: Number of batch
              W: Width of Image : 256
              H: Height of Image: 256
              RGB: 3 Channel
        Output:
        ------
        features_vector Dimension: [N,Feature_Map_size,Feature_Map_size,2048]
        default value of Feature_Map_size is 14
        """
        features_vectors = self.resnet(images)  # [N, 2048, W/32,H/32)
        features_vectors = self.adaptive_pool(features_vectors)  # [N, 2048,Feature_Map_size,Feature_Map_size]      
        features_vectors = features_vectors.permute(0, 2, 3, 1)  # [N,Feature_Map_size,Feature_Map_size,2048] 
        
        return features_vectors

    def finetune(self, fine_tune=True):
        """
        as feature extracter is pretarined model is pretraied model , but in order to improve the accuracy , fine tune can be done. 
        in ResNet 101 block 2 to 4 can be fined tuned,as first layer has basic feature so 1st layers need not to tuned.
        """
        #freeze all the paramatares
        for p in self.resnet.parameters():
            p.requires_grad = False

        # in case of fine tune enable only 2 to 4 layers.
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
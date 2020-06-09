import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib.request import urlopen
from PIL import Image
import numpy as np
import skimage.transform
import torch

#%matplotlib inline

def Visualize_Attention(words,alphas,image=None,image_path=None,image_url=None,smooth=True,image_size=(20,20),scores=None,filename=None):
    """
    input: 
    -----
    alpha       : feature map per pixel 
    words       :
    image_path  : full image path
    image_url   : url of image
    smooth      : boolean value,default is True  

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    """

    if image_path:
        image = Image.open(image_path).convert('RGB')
    elif image_url:
        image = Image.open(urlopen(image_url)).convert('RGB')
    elif isinstance(image,torch.Tensor):
        image = Image.fromarray(np.asarray(image.permute(1,2,0).numpy()*255).astype(np.uint8))
    else:
        print("no imput found for image")
        return
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    plt.figure(figsize=image_size)
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text((14 * 24)//4,(14 * 24)+24, '{0}({1:.2f})'.format(words[t],scores[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t].squeeze(0)
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    plt.show()

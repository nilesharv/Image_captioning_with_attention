import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from urllib.request import urlopen
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerateCaption:
    
    def __init__(self,encoder,decoder,max_length,word_map):
        self.encoder       = encoder
        self.decoder      = decoder
        self.MAX_LENGTH   = max_length
        self.word_map     = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.encoder.eval()
        self.decoder.eval()


    def TranformImage(self,image):
        img=image.resize((256, 256)).convert('RGB')
        img = img=np.asarray(img)
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)

        return image

    def ReadImageFromUrl(self,image_url):
        return Image.open(urlopen(image_url)).convert('RGB')

    def ReadImageFromDisk(self,image_path):
        return Image.open(image_path).convert('RGB')



    def Eval(self,image=None,image_path=None,image_url=None):
    
        if image is None:
            if image_path:
                image = self.ReadImageFromDisk(image_path)
            elif image_url:
                image = self.ReadImageFromUrl(image_url)
            else:
                print("no input found for image")
                return

            image = self.TranformImage(image)
            image = image.unsqueeze(0) 

        image.to(device)

        #######<Encoder>#############
        
        encoder_out = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

        ######</Encoder>#############

        #####<Decoder>###############
        batch_size,enc_image_size, encoder_dim=encoder_out.size(0),encoder_out.size(1),encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(batch_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * batch_size).to(device)  # (k, 1)



         # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)


        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(batch_size, 1).to(device) 

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(batch_size, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        predicted_seq = list()
        predicted_alpha=list()
        predicted_word_scores=list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)
        while True:

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1).to(device)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            scores = F.softmax(scores, dim=1)

            s,word= torch.max(scores,1)
            predicted_word_scores.append(s.item())
            predicted_seq.append(word.item())
            predicted_alpha.append(alpha.cpu().detach().numpy())
            if word == self.word_map['<end>'] or step > self.MAX_LENGTH:
                break
            step+=1
            k_prev_words=word.unsqueeze(0)
        #prediction is done
        #now convert seq to word token and return
        captions=[self.rev_word_map[i] for i in predicted_seq]
        return predicted_alpha,captions,predicted_seq,predicted_word_scores
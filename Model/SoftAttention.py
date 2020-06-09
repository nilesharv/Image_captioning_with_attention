import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Input:
        -----
        encoder_dim   : 2048
        decoder_dim   : 
        attention_dim : 
        """
        super(SoftAttention, self).__init__()

        #linear layer to transform encoder dim to attention dim
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        #linear layer to transform decoder dim to attention dim
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Input:
        -----
        encoder_out(a)    : [N,feature_map*feature_map,encoder_dim]
        decoder_hidden(h) : [N,decoder_dim] 

        Logic:
        -----
        1. Tranform encoder_out(a) to attention_dim using Linear Layers 
            output [N,feature_map*feature_map,attention_dim]
        2. Tranform decoder_hidden(h) to attention_dim using Linear layers 
            output [N,attention_dim] 
        3.  add output of step 1 and 2 
            output [N,feature_map*feature_map,attention_dim]
        4. Apply non linearity RelU on ouput of step 3:
            output [N,feature_map*feature_map,attention_dim]
        5. Tranform output from step 4 to channel 1 using Linear layers 
            output : [N,feature_map*feature_map,1]
        6. Apply softmax over output from step 5 to get Alpha
           output : [N,feature_map*feature_map,1] 
        7. calculate Context weights = sum(a * alpha)
            output : [N,encoder_dim]

        Output:
        -------
        attention_weights : [N,encoder_dim]
        alpha             : [N,feature_map*feature_map,encoder]
        """
        #step 1
        att1 = self.encoder_att(encoder_out)
        
        #step 2
        att2 = self.decoder_att(decoder_hidden)
        #step 3
        sum_a_And_h = att1 + att2.unsqueeze(1)
        #step 4
        relu_out = self.relu(sum_a_And_h)
        #step 5
        att = self.full_att(relu_out).squeeze(2)
        #step 6
        alpha = self.softmax(att)
        #step 7
        context  = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) #this is final context vector sum(alpha * feature vectors)
        
        return context, alpha


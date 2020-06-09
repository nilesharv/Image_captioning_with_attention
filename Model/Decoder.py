import torch
from torch import nn
import torchvision
from .SoftAttention import SoftAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderWithSoftAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        Input:
        ------
        attention_dim   : size of attention network
        embed_dim       : embedding size
        decoder_dim     : Size of Decoder's LSTM
        vocab_size      : Vocabuary Size
        encoder_dim     : Feature vector Size 
        dropout         : droput value for training 
        """
        super(DecoderWithSoftAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Input:
        ------
        encoder_out       : Feature Vectors = [N,feature_map,feature_map,Feature_Vector_Size] 
        encoded_captions  : Encoded captions [N, max_caption_length]
        caption_lengths   : Encoded Captions Lengths [N,1]
        
        logic:
        ------

        Output:
        -------
        predictions       : Predicted Vocab Scores
        encoded_captions  : sorted encoded captions
        decode_lengths    : decode lengths
        alphas            : Alpha Weights 
        sort_ind          : Sorted indecs
        """

        #Extract Dimension
        #N 
        batch_size , encoder_dim, vocab_size = encoder_out.size(0),encoder_out.size(-1),self.vocab_size


        # Flatten image To [N, num_pixels, encoder_dim]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        #store number of pixels : feature_map * feature_map
        num_pixels = encoder_out.size(1)

        #Sort caption and features vector based on Captions Length in Descending, this is to just reduce the overhead during. 
        #and it helps to not pass the <pad> 
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        #do not decode for last word <end> 
        decode_lengths = (caption_lengths - 1).tolist()

        # Create Empty Tensor to store predicted word
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        #Create Empty Tensor to store alpha value crosponding to predicted word
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)


        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)


        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])
            
            hdash , cdash = h[:batch_size_t] ,  c[:batch_size_t] 
            ############<Soft attention >################################
            # 1. get attention weight sum(alpha * feature vectors) 
            # 2. get beta scalar values
            # 3. get vector context = beta * sum(alpha * feature vectors) 
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],hdash)
            gate = self.sigmoid(self.f_beta(hdash))  # gating scalar, [batch_size_t, encoder_dim]
            attention_weighted_encoding = gate * attention_weighted_encoding
            ############</soft attention>###############################

            
            embedding_weights = embeddings[:batch_size_t, t, :]
            
            #merge the embedding and context vector for LSTM
            lstm_input =  torch.cat([embedding_weights, attention_weighted_encoding], dim=1)

            h, c = self.decode_step(lstm_input,(hdash, cdash))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            
            predictions[:batch_size_t, t, :]  =   preds
            alphas[:batch_size_t, t, :]       =   alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

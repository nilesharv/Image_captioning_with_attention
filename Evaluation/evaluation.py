
import torch.backends.cudnn as cudnn
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

from Dataset import *


def GetBlueScore(model,data_loader):
    hypotheses=list()
    references=list()
    word_map = model.word_map
    filter_words ={word_map['<start>'],word_map['<end>'],word_map['<pad>']}
    filter_functon = lambda c: [w for w in c if w not in filter_words]
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(data_loader, desc="EVALUATING on Test Data")):
        image =image.to(device)
        alpha,caption,seq,_ = model.Eval(image)
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        img_caps = allcaps[0].tolist()
        img_captions = list(map(filter_functon,img_caps))   # remove <start> and pads
        references.append(img_captions)
        
    assert len(references) == len(hypotheses)

    #calculate bleu1
    bleu1 = corpus_bleu(references,hypotheses,[1])
    #calculate bleu2
    bleu2 = corpus_bleu(references,hypotheses,[1/2,1/2])
    #calculate bleu3
    bleu3 = corpus_bleu(references,hypotheses,[1/3,1/3,1/3])
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    #calculate bleu5
    bleu5 = corpus_bleu(references,hypotheses,[1/5,1/5,1/5,1/5,1/5])

    return bleu1,bleu2,bleu3,bleu4,bleu5
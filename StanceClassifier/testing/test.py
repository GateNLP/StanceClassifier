import numpy as np
import torch
from scipy.special import softmax


def predict_bertweet(encoded_reply, model):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
  

    encoded_reply = {k: v.to(device) for k, v in encoded_reply.items()}

    with torch.no_grad():
        output_ = model(**encoded_reply) # Output from target-oblivious model, input: encoded_reply
        
    
    logits_ = output_[0][0].detach().cpu().numpy()
    stance_prob, stance_prediction = process_model_output(logits_)

    return stance_prob, stance_prediction

def process_model_output(output_): 
    # input: logits of output_

    #id2label = {0:"support", 1:"deny", 2:"query", 3:"comment"}
    output_ = softmax(output_) # transform logits
    ranking_ = np.argsort(output_)[::-1] # rank
    #return output_[ranking_[0]], id2label[ranking_[0]] 
    return ranking_[0], output_

    
    

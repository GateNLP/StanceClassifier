import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
from scipy.special import softmax


def predict_bertweet(encoded_reply, encoded_source_reply, model_TO, model_TA):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_TO.to(device)
    model_TA.to(device)
    model_TO.eval()
    model_TA.eval()

    encoded_reply = {k: v.to(device) for k, v in encoded_reply.items()}
    encoded_source_reply = {k: v.to(device) for k, v in encoded_source_reply.items()}

    with torch.no_grad():
        output_TO = model_TO(**encoded_reply) # Output from target-oblivious model, input: encoded_reply
        output_TA = model_TA(**encoded_source_reply) # Output from target-aware model, input: encoded_source_reply
    
    #print("output_TO is.............", output_TO)
    #print("output_TA is.............", output_TA)
    logits_TO = output_TO[0][0].detach().cpu().numpy()
    logits_TA = output_TA[0][0].detach().cpu().numpy()
    #print("logits_TO is.............", logits_TO)
    #print("logits_TA is.............", logits_TA)

    #Compare the probability scores from TO and TA models, and take the larger score (and its corresponding label) as the final prediction
    stance_prob, stance_prediction = process_model_output(logits_TO, logits_TA)

    return stance_prob, stance_prediction

def process_model_output(output_TO, output_TA): 
    # input: logits of output_TO and output_TA;

    id2label = {0:"support", 1:"deny", 2:"query", 3:"comment"}
    output_TO = softmax(output_TO) # transform logits
    #print("softmax output_TO is.............", output_TO)
    output_TA = softmax(output_TA)
    #print("softmax output_TA is.............", output_TA)
    
    ranking_TO = np.argsort(output_TO)[::-1] # rank
    #print("ranking_TO is.............", ranking_TO)
    ranking_TA = np.argsort(output_TA)[::-1]
    #print("ranking_TA is.............", ranking_TA)

    #print("output_TA[ranking_TA[0]]", output_TO[ranking_TO[0]], output_TA[ranking_TA[0]])

    if output_TO[ranking_TO[0]] > output_TA[ranking_TA[0]]: # compare
        return output_TO[ranking_TO[0]], id2label[ranking_TO[0]] # probability, label
    else:
        return output_TA[ranking_TA[0]], id2label[ranking_TA[0]]

    
    

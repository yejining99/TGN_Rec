import numpy as np
import torch

class Decoder(torch.nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    pass
    
  def forward(self, user_rep, pos_item_rep, neg_item_rep):

    pos_scores = torch.sum(user_rep * pos_item_rep, dim=1)
    neg_scores = torch.sum(user_rep * neg_item_rep, dim=1)
    
    return pos_scores, neg_scores
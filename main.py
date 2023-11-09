import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
import gc
from tqdm import tqdm
import json
from model.tgn import TGN
from evaluation.evaluation import eval_recommendation
from utils.data import get_data, compute_time_statistics
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
torch.manual_seed(0)
np.random.seed(0)

"""
argument
"""
# setting
parser = argparse.ArgumentParser('TGN recommender training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='transaction')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
# model 
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for each user')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_ngcf", "graph_sum", "identity", "time"], help='Type of embedding module') # jodie, dyrep
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater') # jodie, dyrep
parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model') # dyrep
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message') # dyrep
# training
parser.add_argument('--n_epoch', type=int, default=2, help='Number of epochs')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--num_candidates', type=int, default=3, help='*part of batch items')
parser.add_argument('--num_neg_train', type=int, default=5, help='*p_pos and p_neg items')
parser.add_argument('--test_run', action='store_true', help='*run only first two batches')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
# evaluation
parser.add_argument('--in_sample', action='store_true', help='*Whether to use in-sample setting for evaluation')
parser.add_argument('--num_neg_eval', type=int, default=100, help='*neg items for evaluation')
parser.add_argument('--num_rec', type=int, default=3, help='*top k items for evaluation')
args = parser.parse_args()

"""
global variables
"""
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = 2
DROP_OUT = 0.1
GPU = args.gpu
DATA = args.data # 'Dataset name (eg. wikipedia or reddit)'
NUM_LAYER = 1
LEARNING_RATE = 0.0001
USE_MEMORY = True
MEMORY_DIM = args.memory_dim
MESSAGE_DIM = 100
NUM_CANDIDATES = args.num_candidates
NUM_NEG_TRAIN = args.num_neg_train
NUM_NEG_EVAL = args.num_neg_eval
NUM_REC = args.num_rec
AGGREGATOR = 'last'
MESSAGE_FUNCTION = 'identity'
MEMORY_UPDATE_AT_END = False
MEMORY_UPDATE_AT_START = True
PATIENCE = 5
BACKPROP_EVERY = 1
UNIFORM = False # take uniform sampling from temporal neighbors
USE_SOURCE_EMBEDDING_IN_MESSAGE = False

print(args.prefix)

"""
save paths
"""
Path("results/").mkdir(parents=True, exist_ok=True) # valid, test 결과 저장
Path("saved/").mkdir(parents=True, exist_ok=True) # model checkpoints 저장
get_checkpoint_path = lambda epoch: f'./saved/{args.prefix}_{epoch}.pth'

"""
data
""" 
node_features, edge_features, full_data, train_data, val_data, test_data = get_data(DATA, MEMORY_DIM)
# node_features = np.random.rand(len(node_features), MEMORY_DIM) # memory dim에 맞춰서 node_feautres를 새롭게 랜덤으로 생성하기

"""
init
"""
# Initialize neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, UNIFORM)
full_ngh_finder = get_neighbor_finder(full_data, UNIFORM)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Set device
device = torch.device('cuda:{}'.format(GPU))
gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()

# Initialize Model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=device,
          n_layers=NUM_LAYER,
          n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
          message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
          memory_update_at_start=MEMORY_UPDATE_AT_START,
          embedding_module_type=args.embedding_module,
          message_function=MESSAGE_FUNCTION,
          aggregator_type=AGGREGATOR,
          memory_updater_type=args.memory_updater,
          n_neighbors=NUM_NEIGHBORS,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
          use_destination_embedding_in_message=args.use_destination_embedding_in_message,
          use_source_embedding_in_message=USE_SOURCE_EMBEDDING_IN_MESSAGE,
          dyrep=args.dyrep)

optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

"""
epoch loop
"""
early_stopper = EarlyStopMonitor(max_round=PATIENCE)
best_val_score = 0

for epoch in tqdm(range(NUM_EPOCH), desc="Progress: Epoch Loop" ):  
  start_epoch = time.time()
  
  """
  Train=======================================================================================================================================
  """

  # Reinitialize memory of the model at the start of each epoch
  if USE_MEMORY:
    tgn.memory.__init_memory__()

  # Train using only training graph
  tgn.set_neighbor_finder(train_ngh_finder)

  """
  batch loop
  """
  losses_batch = []

  for batch in tqdm(range(0, num_batch, BACKPROP_EVERY), total=num_batch//BACKPROP_EVERY, desc="Progress: Train Batch Loop"):

    # test run
    if args.test_run:
      if batch == 2:
        break

    loss = 0
    optimizer.zero_grad()

    # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(BACKPROP_EVERY):

      batch_idx = batch + j

      if batch_idx >= num_batch:
        continue

      s_idx = batch_idx * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)
      
      # batch data 뽑기: <class 'numpy.ndarray'>
      sources_batch = train_data.sources[s_idx:e_idx]           # (BATCH_SIZE,)
      destinations_batch = train_data.destinations[s_idx:e_idx] # (BATCH_SIZE,) # item idx
      edge_idxs_batch  = train_data.edge_idxs[s_idx: e_idx]     # (BATCH_SIZE,)
      timestamps_batch = train_data.timestamps[s_idx:e_idx]     # (BATCH_SIZE,)
      
      # candidate sampling
      train_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch)
      negative_batch = train_rand_sampler.sample(size=NUM_CANDIDATES)  # (BATCH_SIZE, size) # item idx
      
      # flatten negative_batch
      negative_batch = np.array([x for y in negative_batch for x in y])

      """
      emb 계산
      """
      tgn = tgn.train()
      source_embedding, destination_embedding, neg_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                                                              destinations_batch,
                                                                                              negative_batch,
                                                                                              timestamps_batch,
                                                                                              edge_idxs_batch,
                                                                                              NUM_NEIGHBORS)

      """
      loss 계산
      """

      bsbs = source_embedding.shape[0]

      # reshape source and destination to (bs, 1, emb_dim) 
      source_embedding = source_embedding.view(bsbs, 1, -1)
      destination_embedding = destination_embedding.view(bsbs, 1, -1)

      # reshape p_pos and p_neg to (bs, k, emb_dim) 
      neg_embedding = neg_embedding.view(bsbs, NUM_CANDIDATES, -1)

      # BPR loss
      pos_scores = torch.sum(source_embedding * destination_embedding, dim=2)                             # (bsbs, 1)
      neg_scores = torch.matmul(source_embedding, neg_embedding.transpose(1, 2)).squeeze()              # (bsbs, k)

      # print('pos_scores: ', pos_scores[:3])
      # print('neg_scores: ', neg_scores[:3])
      # print('==')

      score_diff = pos_scores - neg_scores                                                                # (bsbs, k)
      score_diff_mean = torch.mean(score_diff, dim=1)                                                     # (bsbs, )
      log_and_sigmoid = torch.log(torch.sigmoid(score_diff_mean))                                         # (bsbs, )
      loss_BPR = -torch.mean(log_and_sigmoid)                                                             # (1, )

      loss += loss_BPR
      
    loss /= BACKPROP_EVERY
    loss.backward()
    optimizer.step()
    losses_batch.append(loss.item())

    # Detach memory after 'BACKPROP_EVERY' number of batches so we don't backpropagate to the start of time
    if USE_MEMORY:
      tgn.memory.detach_memory()

  torch.save(tgn.state_dict(), get_checkpoint_path(epoch))  

  """
  Valid=======================================================================================================================================
  """
  # Validation uses the full graph
  tgn.set_neighbor_finder(full_ngh_finder)

  eval_dict = eval_recommendation(tgn=tgn,
                                  data=val_data, 
                                  batch_size=BATCH_SIZE,
                                  n_neighbors=NUM_NEIGHBORS,
                                  NUM_NEG_EVAL = NUM_NEG_EVAL,
                                  is_test_run=args.test_run)
  
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  """
  An epoch done, save results to disk
  """

  # 추천 성능
  num_topk = len(eval_dict['recalls'][0]) # e.g., top 1,5,10,20 -> 4
  recalls = eval_dict['recalls']      # e.g., [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], ...]
  ndcgs = eval_dict['ndcgs']
  mrrs = eval_dict['mrrs']
  hits = eval_dict['hits']
  precisions = eval_dict['precisions']
  recall_avg = np.mean([recalls[i][2] for i in range(len(recalls))]) # top10 only
  ndcg_avg = np.mean([ndcgs[i][2] for i in range(len(ndcgs))])
  mrr_avg = np.mean([mrrs[i][2] for i in range(len(mrrs))])
  hit_avg = np.mean([hits[i][2] for i in range(len(hits))])
  precision_avg = np.mean([precisions[i][2] for i in range(len(precisions))])


  val_dict = {'val_recall_avg': recall_avg, 
              'val_ndcg_avg': ndcg_avg, 
              'val_mrr_avg': mrr_avg,
              'val_hit_avg': hit_avg,
              'val_precision_avg': precision_avg,
              }
  
  


  results_path = f'./results/{args.prefix}_valid.pkl'
  pickle.dump(eval_dict, open(results_path, 'wb'))

  """
  save checkpoint (and early stopping)
  """

  # current_val_score = recall_avg
  # if early_stopper.early_stop_check(current_val_score):
  #   # logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
  #   # logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
  #   wandb.log({'early stop': early_stopper.best_epoch})
  #   best_model_path = get_checkpoint_path(early_stopper.best_epoch)
  #   tgn.load_state_dict(torch.load(best_model_path))
  #   tgn.eval()
  #   break
  # else:
  #   torch.save(tgn.state_dict(), get_checkpoint_path(epoch))       
  #   wandb.save(get_checkpoint_path(epoch)) 

  torch.save(tgn.state_dict(), get_checkpoint_path(epoch))       


  """
  epoch loop done
  """
  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  """
  Test=======================================================================================================================================
  """
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  eval_dict_test = eval_recommendation(tgn,
                                        test_data, 
                                        BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS,
                                        NUM_NEG_EVAL = NUM_NEG_EVAL,
                                        is_test_run=args.test_run,
                                      )


  """
  Save results for this run
  """

  # 추천 성능
  num_topk = len(eval_dict_test['recalls'][0]) # e.g., top 1,5,10,20 -> 4
  recalls = eval_dict_test['recalls']      # e.g., [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], ...]
  ndcgs = eval_dict_test['ndcgs']
  mrrs = eval_dict_test['mrrs']
  hits = eval_dict_test['hits']
  precisions = eval_dict_test['precisions']
  recall_avg = np.mean([recalls[i][2] for i in range(len(recalls))]) # top10 only
  ndcg_avg = np.mean([ndcgs[i][2] for i in range(len(ndcgs))])
  mrr_avg = np.mean([mrrs[i][2] for i in range(len(mrrs))])
  hit_avg = np.mean([hits[i][2] for i in range(len(hits))])
  precision_avg = np.mean([precisions[i][2] for i in range(len(precisions))])

  test_dict = {'test_recall_avg': recall_avg, 
                'test_ndcg_avg': ndcg_avg, 
                'test_mrr_avg': mrr_avg,
                'test_hit_avg': hit_avg,
                'test_precision_avg': precision_avg,
                }

  results_path = f'./results/{args.prefix}_test.pkl'
  pickle.dump(eval_dict_test, open(results_path, 'wb'))


  print('Saving TGN model')

  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), f'./saved/{args.prefix}_test.pth')

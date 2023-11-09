import math
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder


def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))

def ndcg_at_k(recommendations, test_items, k):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_items)))])
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg

def MRR_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1 / (i + 1)
    return 0

def Hit_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1
    return 0
  
def Precision_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / k

def eval_recommendation(tgn, data, batch_size, n_neighbors, NUM_NEG_EVAL, is_test_run):
    with torch.no_grad():
        tgn = tgn.eval()
        # While usually the test batch size is as big as it fits in memory, 
        # here we keep it the same size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        """
        batch iteraction
        """
        recalls, ndcgs, mrrs, hits, precisions = [], [], [], [], []
          
        for batch in tqdm(range(num_test_batch), desc=f"Progress: Eval Batch"):
        # for batch in range(num_test_batch):

          s_idx = batch * TEST_BATCH_SIZE
          e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

          # 마지막 배치는 건너뛴다
          if e_idx == num_test_instance:
            continue

          # test run
          if is_test_run:
            if batch == 2:
              break

          # batch data 뽑기: <class 'numpy.ndarray'>
          sources_batch = data.sources[s_idx:e_idx]           # (BATCH_SIZE,)
          destinations_batch = data.destinations[s_idx:e_idx] # (BATCH_SIZE,) # item idx
          timestamps_batch = data.timestamps[s_idx:e_idx]     # (BATCH_SIZE,)
          edge_idxs_batch = data.edge_idxs[s_idx: e_idx]      # (BATCH_SIZE,)

          # negative sampling
          test_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch, seed=2023)
          negatives_batch = test_rand_sampler.sample(size=NUM_NEG_EVAL) # (BATCH_SIZE, size) # item idx

          """
          node embedding 생성
          """
          source_embedding, destination_embedding, negative_embedding = tgn.compute_temporal_embeddings_eval(sources_batch,
                                                                                                            destinations_batch,
                                                                                                            negatives_batch.flatten(), # (BATCH_SIZE * size,)
                                                                                                            timestamps_batch,
                                                                                                            edge_idxs_batch,
                                                                                                            n_neighbors)
          """
          score 계산
          """

          bsbs = source_embedding.shape[0] # 마지막 배치는 겨우 23개.. 이 경우 neg item size가 일정하지 않다 

          # reshape source and destination to (bs, 1, emb_dim) 
          source_embedding = source_embedding.view(bsbs, 1, -1)
          destination_embedding = destination_embedding.view(bsbs, 1, -1)

          # reshape negative to (bs, size, emb_dim)
          negative_embedding = negative_embedding.view(bsbs, NUM_NEG_EVAL, -1)

          # scores ( <class 'numpy.ndarray'> )
          pos_scores = torch.sum(source_embedding * destination_embedding, dim=2).cpu().numpy() # (bs, 1)
          neg_scores = torch.sum(source_embedding * negative_embedding, dim=2).cpu().numpy()    # (bs, size)
          
          """
          interaction loop
          """
          for i in range(bsbs):

            """
            추천 평가
            """

            # score 통해서 ranking 구하기
            pos_score = pos_scores[i] # pos score 한 개   <class 'numpy.ndarray'>  (1,)
            neg_score = neg_scores[i] # neg score size개  <class 'numpy.ndarray'>  (100,)

            scores = np.concatenate((pos_score, neg_score))  # [0.09, 0.88, 0.22, 0.15]
            ranking = np.argsort(scores)[::-1]               # [1, 2, 3, 0]

            # recall, ndcg 구하기
            pos_ranking = [0] # ranking에서 pos item의 위치는 항상 0 # 만약 pos item이 여러 개면 pos_ranking = [0, 1, 2, ..., len_pos_item-1]]
            topk = [1, 5, 10, 20]
            recall = [recall_at_k(ranking, pos_ranking, top) for top in topk]
            ndcg = [ndcg_at_k(ranking, pos_ranking, top) for top in topk]
            mrr = [MRR_at_k(ranking, pos_ranking, top) for top in topk]
            hit = [Hit_at_k(ranking, pos_ranking, top) for top in topk]
            precision = [Precision_at_k(ranking, pos_ranking, top) for top in topk]

            """
            store results
            """
            # list len = num_test_batch
            recalls.append(recall)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            hits.append(hit)
            precisions.append(precision)

        eval_dict = {'recalls': recalls,
                    'ndcgs': ndcgs,
                    'mrrs': mrrs,
                    'hits': hits,
                    'precisions': precisions
                    }

        return eval_dict 

    
    
    
    
    
    # source_nodes = data.sources
    # destination_nodes = data.destinations
    # size = len(source_nodes)
    # _, negative_nodes = negative_edge_sampler.sample(size)
    # edge_times = data.timestamps
    # edge_idxs = data.edge_idxs
    
    # """
    # node embedding 생성
    # """
    # source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(source_nodes,
    #                                                                               destination_nodes,
    #                                                                               negative_nodes,
    #                                                                               edge_times,
    #                                                                               edge_idxs,
    #                                                                               n_neighbors)
    
    # """
    # 유저마다 user_purchase_history 생성
    # """
    
    # # create a dict 'user_buy_dict' where keys are unique source_nodes and values are lists of destination_nodes that the source_nodes have purchased
    # source_nodes_set = np.unique(source_nodes)
    # destination_nodes_set = np.unique(destination_nodes)
    # user_buy_dict = {source_node: destination_nodes[source_nodes == source_node] for source_node in source_nodes_set}
    
    # """
    # 유저 loop 돌면서 평가
    # """
    
    # # print('Evaluation Start, num of users: ', len(user_buy_dict), len(source_nodes_set))
    # # print('Evaluation Start, num of items: ', len(destination_nodes_set))
    # sum_recall = 0.0
    # sum_ndcg = 0.0
    # sum_mrr = 0.0
    # sum_hit = 0.0
    # sum_precision = 0.0
    # total_user = 0
    
    # for user, pos_items in user_buy_dict.items():
      
    #   """
    #   예시
    #   user:  1                                                      # numpy.int64
    #   pos_items:  [274 274 274 274 274 274 274 274 274 274 274 274] # numpy.ndarray
    #   neg_items = [517]                                             # numpy.ndarray
    #   """
      
    #   # pos_items 없는 유저는 평가에서 제외
    #   if len(pos_items) == 0:
    #     continue
      
    #   neg_items = np.setdiff1d(destination_nodes_set, pos_items)
    #   # neg_items 100개 미만인 유저는 평가에서 제외
    #   if len(neg_items) < 100:
    #     continue
    #   neg_items = random.sample(list(neg_items), 100)
      
    #   user_tensor = torch.LongTensor([user]).to(tgn.device)
    #   pos_tensor = torch.LongTensor(pos_items).to(tgn.device)
    #   neg_tensor = torch.LongTensor(neg_items).to(tgn.device)
      
    #   user_emb = source_embedding[user_tensor]
    #   pos_emb = destination_embedding[pos_tensor]
    #   neg_emb = destination_embedding[neg_tensor]
      
    #   pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    #   neg_scores = torch.sum(user_emb * neg_emb, dim=1)
      
    #   k = 10
    #   ranking = torch.argsort(torch.cat([pos_scores.flatten(), neg_scores.flatten()]), descending=True).cpu().numpy().tolist()
    #   pos_ranking = [i for i in range(len(pos_scores))]
      
    #   recall = recall_at_k(ranking, pos_ranking, k)
    #   ndcg = ndcg_at_k(ranking, pos_ranking, k)
    #   mrr = MRR_at_k(ranking, pos_ranking, k)
    #   hit = Hit_at_k(ranking, pos_ranking, k)
    #   precision = Precision_at_k(ranking, pos_ranking, k)

    #   sum_recall += recall   
    #   sum_ndcg += ndcg
    #   sum_mrr += mrr
    #   sum_hit += hit
    #   sum_precision += precision
    #   total_user += 1
      
    # return sum_recall/total_user, sum_ndcg/total_user, sum_mrr/total_user, sum_hit/total_user, sum_precision/total_user
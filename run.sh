
for DATASET in ml_lm netflix yelp; do
    for GNN in graph_attention graph_NGCF graph_sum; do
        for RNN in gru rnn; do
            echo "Running with GNN=$GNN and RNN=$RNN and DATASET=$DATASET"
            python main.py --use_memory --memory_updater $RNN --embedding_module $GNN --prefix ${GNN}_${RNN}_${DATASET} --data $DATASET ;
        done
    done
done


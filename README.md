# Temporal Graph Networks for Recommender Systems (TGN-RS)

TGN-RS is an advanced recommender system framework that leverages Temporal Graph Networks to provide accurate and time-sensitive recommendations. This model integrates user-item interactions over time, capturing temporal dynamics for improved recommendation performance.

## Dataset

The TGN Recommender System requires a dataset to function properly. You can download the dataset from the following link:

[Download Dataset](https://unistackr0-my.sharepoint.com/:f:/g/personal/kimyejin99_unist_ac_kr/EgHSAM0-ikxGsiRsQwrJm5kBGKrgOhP7AxJyOqNpDZzJlQ?e=0KgBqg)

After downloading, place the dataset in the `data` folder at the root of the project directory.

## Running the Code

To run the Temporal Graph Networks (TGN) for Recommender Systems, you will need to execute the `main.py` script with the necessary command-line arguments.

### Basic Usage

At the most basic level, you can run the code without any additional arguments (which will use the default settings):

```bash
python main.py
```

### Advanced Usage

For a more advanced setup, you can specify various parameters to customize the training. Here's an example command that utilizes memory, sets the memory updater, the embedding module, and a prefix for saving checkpoints:

```bash
python main.py --use_memory --memory_updater gru --embedding_module graph_attention --prefix graph_attention_gru_wikipedia --data wikipedia
```

Replace gru with the type of RNN you wish to use for the memory updater (e.g., 'rnn', 'lstm'), graph_attention with the type of GNN for the embedding module, and wikipedia with the name of your dataset.


## Structure

The codebase is organized into several key components:

- `evaluation/`: Contains `evaluation.py` for model performance assessment.
- `model/`: Houses model definitions including `decoder.py`, `temporal_attention.py`, `tgn.py`, and `time_encoding.py`.
- `module/`: Includes modules such as `embedding_module.py`, `memory.py`, `memory_updater.py`, `message_aggregator.py`, and `message_function.py`.
- `utils/`: Provides utility scripts like `data.py`, `preprocess_data.py`, and `utils.py`.
- `main.py`: The main script to run the TGN-RS model.
- `run.sh`: A shell script to execute the model with predefined settings.

## Model Configuration

The Temporal Graph Networks (TGN) for Recommender Systems is configured through a set of command-line arguments. Here are the details of the configurations that can be adjusted:

### Setting
- `--data`: Dataset name (e.g., 'wikipedia', 'reddit', 'transaction') - the dataset to use.
- `--gpu`: Index of the GPU to use.
- `--prefix`: Prefix to name the checkpoints.

### Model Parameters
- `--memory_dim`: Dimension of the memory for each user (default: 64).
- `--n_degree`: Number of neighbors to sample (default: 10).
- `--embedding_module`: Type of embedding module, options include "graph_attention", "graph_ngcf", "graph_sum", "identity", "time".
- `--memory_updater`: Type of memory updater, options include "gru", "rnn".
- `--dyrep`: Flag to run the dyrep model.
- `--use_destination_embedding_in_message`: Flag to use the destination node's embedding as part of the message.

### Training Parameters
- `--n_epoch`: Number of epochs (default: 2).
- `--bs`: Batch size (default: 1000).
- `--num_candidates`: Part of batch items (default: 3).
- `--num_neg_train`: p_pos and p_neg items (default: 5).
- `--test_run`: Flag to run only the first two batches for testing.
- `--use_memory`: Flag to augment the model with a node memory.

### Evaluation Parameters
- `--in_sample`: Flag to use in-sample setting for evaluation.
- `--num_neg_eval`: Negative items for evaluation (default: 100).
- `--num_rec`: Top-k items for evaluation (default: 3).

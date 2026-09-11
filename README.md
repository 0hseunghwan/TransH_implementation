# TransH Link Prediction (FB15k-237)

This is a PyTorch reimplementation of the Link Prediction experiment from the TransH paper (Wang, Zhang, Feng, Chen, *"Knowledge Graph Embedding by Translating on Hyperplanes"*, AAAI 2014), one of three experiments described in the paper.

## What's Implemented

- Learns a hyperplane (normal vector `w_r`) and a translation `d_r` on that hyperplane for each relation
- Projects entities onto the relation-specific hyperplane, then scores them the same way as TransE
- Adds soft constraints (entity norm limit, `w_r`-`d_r` orthogonality) to the loss as regularization terms
- Evaluation metrics: Mean Rank (MR), MRR, Hits@10 (raw / filtered)

## File Structure

```
.
├── TransH.ipynb   # main notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

GPU (CUDA) is used automatically if available. If running on Google Colab,
it's recommended to set `Runtime > Change runtime type > GPU`.

### 2. Run the Notebook

Run `TransH.ipynb` in order:

1. **Environment setup** — install torch and check device
2. **Dataset download** — automatically downloads FB15k-237 (`train.txt`, `valid.txt`, `test.txt`) from the
   [dataset_FB15k-237](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237)
   repository and stores it in the `fb15k237/` directory
3. **Data loading and indexing** — map entities/relations to integer IDs
4. **TransH model implementation** — including hyperplane projection, score function, and soft constraints
5. **Negative sampling** — randomly replace head or tail (uniform)
6. **Training** — margin ranking loss + SGD
7. **Training loss visualization**
8. **Evaluation** — MR / MRR / Hits@10 (raw, filtered)

## Key Hyperparameters

The defaults are set for quick experimentation and may differ from the hyperparameter values used in the paper.

| Parameter | Default | Description |
|---|---|---|
| `DIM` | 100 | Embedding dimension |
| `BATCH_SIZE` | 1024 | Batch size |
| `EPOCHS` | 20 | 500+ epochs recommended to reproduce the paper's results |
| `LR` | 0.01 | SGD learning rate |
| `MARGIN` | 1.0 | Margin for margin ranking loss |
| `C` | 0.25 | Soft constraint weight |


## Notes / Tuning Tips

- Refer to the TransE / TransH papers for `EPOCHS`, `DIM`, `LR`, `MARGIN`, and `C` (soft constraint weight)
- Increasing `max_test` in `evaluate` (e.g., to `None` for the full test set) reproduces the paper's full evaluation,
  but requires a forward pass for every entity × test triple combination, which is very slow on CPU. Using a GPU runtime is recommended
  (`Runtime > Change runtime type > GPU`)
- Negative sampling currently uses uniform random. Switching to the paper's "bern" strategy
  (probabilistically replacing head/tail based on the average number of tails/heads per head/tail for a relation) may improve performance further.

## References

Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). *Knowledge Graph Embedding by Translating on Hyperplanes.*
Proceedings of the AAAI Conference on Artificial Intelligence, 28(1).

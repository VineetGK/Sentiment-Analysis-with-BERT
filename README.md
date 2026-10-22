# Sentiment Analysis with BERT

Fine-tunes `bert-base-cased` to classify Google Play app reviews as **negative, neutral or positive** (3 classes).
Everything is in `Sentiment Analysis with BERT (1).ipynb`.

This notebook closely follows Venelin Valkov's tutorial "Sentiment Analysis with BERT and Hugging Face" (from his *Getting Things Done with PyTorch* series).

## Data

- `reviews.csv`: 15,746 Google Play reviews of productivity apps (columns include `content`, `score`, `appId`).
- **The CSV is not included in this repo.** Place a `reviews.csv` with at least `content` and `score` columns next to the notebook.
- Labels come from the star rating: 1-2 = negative, 3 = neutral, 4-5 = positive.
- Split: 90% train, 5% validation, 5% test (14,171 / 787 / 788 rows).

## Model and training

- `BertTokenizer` (`bert-base-cased`), max length 160 tokens.
- `SentimentClassifier`: `BertModel` pooled output, then dropout (p=0.3), then a linear layer with 3 outputs.
- AdamW (lr 2e-5), linear schedule with no warmup, gradient clipping at 1.0, batch size 16, 10 epochs, cross-entropy loss.
- The weights with the best validation accuracy are saved to `best_model_state.bin`.

## How to run

The notebook was run on a GPU (Colab, Python 3.7).

1. `pip install -r requirements.txt` (note the old `transformers` version, see below).
2. Put `reviews.csv` next to the notebook.
3. Before running, define the two missing variables (see Known limitations), for example `RANDOM_SEED = 42` and a `sample_txt` string.
4. Run all cells. Training took about 44 minutes on the GPU used.

## Results (from the committed outputs)

| | Accuracy |
|---|---|
| Validation, epoch 1 | 0.778 |
| Validation, epoch 10 (best) | 0.893 |
| **Test** | **0.883** |

Test classification report (788 reviews):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.89 | 0.87 | 0.88 |
| neutral | 0.83 | 0.85 | 0.84 |
| positive | 0.92 | 0.93 | 0.92 |

Neutral reviews are the hardest class.

## Known limitations

- The notebook does not run top to bottom as committed: `RANDOM_SEED` (used in the split cell) and `sample_txt` (used in the tokenizer demo) are never defined, and `reviews.csv` is not in the repo.
- Needs `transformers` 3.x or older. On v4+ it breaks: the model output is tuple-unpacked (`_, pooled_output = self.bert(...)`), `transformers.AdamW` (with `correct_bias`) has been removed, and `pad_to_max_length` is deprecated.
- The test score uses the model in memory after the last epoch, not the saved `best_model_state.bin`. In this run the last epoch was also the best on validation, but that is not guaranteed.
- The model overfits: training accuracy reaches 0.988 while validation loss rises from 0.51 (epoch 3) to 0.74 (epoch 10).
- No package versions were recorded, and the run is not fully seeded.

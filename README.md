# Hausa News Topic Classification

This project classifies Hausa news articles into seven topics: business, entertainment, health, politics, religion, sports and technology. It replicates the baseline experiments of the MasakhaNEWS paper, extends them with three encoding methods, and serves the best model through a web interface.

## Project overview

1. **Replication.** The classical baselines from the paper (Naive Bayes, MLP and XGBoost on word counts) were rebuilt and run on the Hausa data.
2. **Extension.** Three encoding methods were tested: c-TF-IDF, Model2Vec (a compressed LaBSE model) and Potion (a multilingual Model2Vec model).
3. **Classifiers.** Each encoder was paired with a linear SVM and with FastKAN, a neural network built on Kolmogorov-Arnold layers.
4. **Evaluation.** All models were scored on the same held-out test set using weighted F1, and compared with the paper's published Hausa results.
5. **Web app.** The best model (Potion with SVM) is served through a Gradio interface that predicts the topic of any pasted Hausa text.

## Dataset

The data is the Hausa subset of MasakhaNEWS, loaded from Hugging Face as `masakhane/masakhanews` (config `hau`).

| Split      | Articles |
|------------|---------:|
| Train      | 2,219    |
| Validation | 317      |
| Test       | 637      |

Each article has a headline and a body text. The model input is the headline and body joined together. The seven topics are business, entertainment, health, politics, religion, sports and technology.

## Results

Weighted F1 on the test set (higher is better). Replication results are averaged over five seeds, and so are the FastKAN results.

| Model                        | Weighted F1 | Paper (Hausa) |
|------------------------------|------------:|--------------:|
| Naive Bayes (replication)    | 87.2        | 85.3          |
| MLP (replication)            | 88.5        | 86.7          |
| XGBoost (replication)        | 83.7        | 84.7          |
| Potion + SVM                 | **85.7**    | not reported  |
| Potion + FastKAN             | 82.7        | not reported  |
| Model2Vec + SVM              | 84.0        | not reported  |
| Model2Vec + FastKAN          | 82.9        | not reported  |
| c-TF-IDF + SVM               | 55.5        | not reported  |
| c-TF-IDF + FastKAN           | 65.3        | not reported  |

The paper's Hausa results for fine-tuned models were AfroXLMR-base 91.0, AfroXLMR-large 92.2 and LaBSE 90.0. These were not reproduced in this project.

The full results table is saved in `results.csv`.

## Method notes

- **c-TF-IDF:** A simplified version. Word weights are calculated per topic, averaged into one weight per word, and applied to a vocabulary capped at 5,000 words. This simplification is the most likely reason for the low scores.
- **Model2Vec:** The LaBSE sentence model is compressed to 256 dimensions with `model2vec`.
- **Potion:** The pretrained multilingual model `minishlab/potion-multilingual-128M`, producing 256-dimensional vectors.
- **SVM:** `LinearSVC` with default settings.
- **FastKAN:** One hidden layer of 64 units, trained for 30 epochs with the Adam optimiser (learning rate 0.001) and batch size 32. These settings were not tuned.
- **Replication settings:** The paper does not list its hyperparameters, so scikit-learn defaults were used. This explains small differences from the paper's numbers.

## Repository contents

| File | Description |
|------|-------------|
| `HausaTopic_Project.ipynb` | Complete Colab notebook: data loading, replication, encoders, classifiers, evaluation and the Gradio app |
| `results.csv` | Weighted F1 scores for all models |
| `Hausa_Topic_Classification_Guide.docx` | Step-by-step guide to reproducing the project |
| `Hausa_Topic_Classification_Proposal.docx` | One-page project proposal |

## How to run

The project was built and tested in Google Colab on a CPU runtime.

1. Open `HausaTopic_Project.ipynb` in Google Colab (File, then Open notebook, then GitHub).
2. Run the install cells, including `pip install "model2vec[distill]"` and the FastKAN install from GitHub:
!pip install -q git+https://github.com/ZiyaoLi/fast-kan.git

3. Run the cells in order. The Model2Vec encoding step takes about an hour on CPU.
4. The final cell starts the Gradio app and prints a public link. Public links are temporary.

Colab sessions can restart, which clears the variables in memory. The notebook saves the features to files (`.npy` and `.npz`) so that they can be reloaded instead of recomputed.

## Limitations

- The test set has 637 articles, so differences of one or two F1 points may not be significant.
- FastKAN was not tuned, and the c-TF-IDF encoder uses a simplified weighting, so these results should not be taken as the best achievable with these methods.
- FasterKAN, one of the classifiers listed in the project brief, was not used. SVM and FastKAN were chosen instead.
- The paper's fine-tuned transformer baselines (AfroXLMR, LaBSE) were not reproduced, apart from the optional step described in the guide.

## Reference

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J. O., Tonja, A. L., Mwase, C., ... & Stenetorp, P. (2023). MasakhaNEWS: News Topic Classification for African languages. *Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 144–159.

## Author

Kafilat Musa, MSc Artificial Intelligence, Ahmadu Bello University, Zaria.

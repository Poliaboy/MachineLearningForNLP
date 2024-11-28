# Report: Enhancing Recommendation Systems with NLP
##### By Alex Szpakiewicz and Léonard Roussard

## Project Overview

This project aimed to develop a recommendation system capable of identifying the most semantically similar hotel based on user reviews. The objective was to surpass BM25 without preprocessing—a foundational information retrieval model. By exploring hybrid approaches, transformer-based architectures, and specialized models, the goal was to improve both accuracy and efficiency while addressing BM25’s inherent limitations.

## Introduction

The dataset used was the **TripAdvisor Hotel Review dataset**, which includes textual reviews and ratings for seven aspects: service, cleanliness, overall experience, value, location, sleep quality, and rooms. To prepare the dataset:

1. **Filtering**: Reviews missing ratings for any of the seven aspects were removed.
2. **Aggregation**: Reviews were grouped by the `offering_id` attribute, consolidating all textual data for each hotel.
3. **Averaging**: Aspect ratings were averaged for each hotel to create a unified metric for evaluation.

Using my MacBook Pro with an M3 Pro CPU and GPU, I was able to process the entire dataset efficiently. Unlike in cloud environments like Colab, where performance limitations often restrict data usage, my setup enabled comprehensive experimentation without compromising speed.

## BM25 Baseline Evaluation

### What is BM25?

BM25 is a ranking function based on keyword matching between a query and documents. It evaluates:

1. **Term Frequency (TF)**: Importance of a word within a document.
2. **Inverse Document Frequency (IDF)**: Rarity of the word across the entire corpus.
3. **Length Normalization**: Preference for shorter documents when scores are otherwise identical.

BM25 is simple yet effective, making it a common baseline for information retrieval tasks.

### Results

BM25 was tested in two scenarios:

1. **Without Preprocessing (Baseline for Comparison)**: Raw reviews were used.
   - Average Scoring Time per Query: **26.83 seconds**
   - Total Time: **~44.72 minutes**
   - MSE: **0.5350**

2. **With Preprocessing**: Tokenization, stop-word removal, and text cleaning were applied as a custom improvement.
   - Average Scoring Time per Query: **11.88 seconds**
   - Total Time: **~19.80 minutes**
   - MSE: **0.4734**

While BM25 with preprocessing demonstrated notable improvements, the primary baseline for comparison in this project was BM25 without preprocessing. Notably, all other models were implemented on the cleaned dataset, showcasing BM25’s robustness against more complex models.

## Approach to Model Exploration

The iterative approach started with BM25 and sought to address its limitations. BM25 without preprocessing, while effective, had a high MSE and slow processing speed. My first improvement involved combining BM25 with semantic embeddings to better capture meaning. However, the hybrid model still relied on BM25’s CPU-bound processing, limiting speed.

Subsequent efforts focused on models capable of leveraging GPU acceleration to significantly reduce query time. These included transformer-based architectures like MPNet and sentence comparison models like SimSec. Finally, I tested the state-of-the-art **MXBI Colbert**, which combines advanced semantic understanding with efficient retrieval mechanisms, achieving promising results.

## Process and Model Analysis

### Hybrid Model

The hybrid model combined BM25’s keyword-based scoring with semantic embeddings from a pre-trained model. This approach aimed to enhance semantic understanding while retaining BM25’s precision in lexical matching.

- **Performance**: MSE of **0.4910**, surpassing BM25 without preprocessing but slightly worse than BM25 with preprocessing.
- **Efficiency**: Average query time of **12.84 seconds**; total processing time of **~21.40 minutes**.
- **Limitations**: The reliance on BM25 meant the hybrid model was CPU-bound, significantly impacting speed.
- **Analysis**: While this model demonstrated meaningful improvement in accuracy, its speed limitations made it impractical for large-scale applications.

### Transformer-Based Model: MPNet

MPNet is a transformer-based architecture designed for contextual embedding generation, capturing nuanced relationships between words.

- **Performance**: MSE of **0.5476**, slightly worse than BM25 without preprocessing.
- **Efficiency**: Leveraged GPU acceleration, achieving an average query time of **0.05 seconds**.
- **Analysis**: MPNet offered significant speed advantages due to its GPU support but fell short in accuracy.

### SimSec Model

The SimSec model focuses on lightweight sentence similarity, offering a simpler architecture compared to transformers.

- **Performance**: MSE of **0.5574**, worse than MPNet and BM25 without preprocessing.
- **Efficiency**: Fastest model tested, with an average query time of **0.04 seconds**.
- **Analysis**: While highly efficient, SimSec’s accuracy was insufficient for this application.

### MXBI Colbert

#### Results

- **MSE**: **0.4954**, better than BM25 without preprocessing but slightly worse than BM25 with preprocessing.
- **Average Query Time**: **0.12 seconds**
- **Total Processing Time**: **~0.19 minutes**

#### Architecture

MXBI Colbert combines token-level embeddings with advanced retrieval techniques:

1. **Token-Level Contextualization**: Transformer layers generate detailed embeddings for each token.
2. **Late Interaction**: Enables direct token-to-token comparisons, enhancing precision.
3. **Pooling and Normalization**: Mean pooling is applied, and embeddings are L2-normalized.

#### Implementation

Steps executed using the Hugging Face Transformers library:

1. **Loading Pretrained Components**: The `mixedbread-ai/mxbai-colbert-large-v1` model and tokenizer were initialized.
2. **Embedding Generation**: Reviews were tokenized and passed through transformer layers to generate contextualized embeddings.
3. **Normalization**: Embeddings were normalized using L2 normalization.
4. **Query Matching**: Cosine similarity between query and document embeddings was calculated.
5. **Optimization**: A softmax function with temperature scaling was applied to similarity scores.

### Comparative Results

The models, sorted by MSE, are summarized below:

| Model                 | Avg Scoring Time (s) | Total Time (min) | MSE    |
|-----------------------|----------------------|------------------|--------|
| BM25 (with preprocess)| 11.88               | 19.80            | 0.4734 |
| Hybrid Model          | 12.84               | 21.40            | 0.4910 |
| MXBI Colbert          | 0.12                | 0.19             | 0.4954 |
| BM25 (no preprocess)  | 26.83               | 44.72            | 0.5350 |
| MPNet                 | 0.05                | 0.09             | 0.5476 |
| SimSec Model          | 0.04                | 0.07             | 0.5574 |

## Conclusion

BM25 without preprocessing, despite its simplicity, served as a strong baseline. Among the tested models:

- The hybrid model demonstrated better accuracy but was hindered by CPU-bound limitations.
- MXBI Colbert achieved a balance of accuracy and speed, leveraging GPU acceleration for efficient semantic retrieval.
- BM25 with preprocessing achieved the best results overall, highlighting the importance of text cleaning.

## Insights and Learning Outcomes

This project highlighted the surprising efficiency of BM25 for retrieval tasks, even when compared to state-of-the-art architectures. GPU-accelerated BM25 implementations provide a promising direction to overcome CPU-bound limitations. Further training on this dataset would yield excellent results but requires high resources, making it less practical for real-world applications.

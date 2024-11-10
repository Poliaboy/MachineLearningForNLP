# Information Retrieval Models Comparison
**Authors**: Alex Szpakiewicz, Léonard roussard 
**Date**: 10/11/2024
**Colab Notebook**: [Your Colab Link]

## Introduction
This report presents our exploration of various information retrieval models, comparing their performance against a baseline BM25 model. We implemented and evaluated several approaches, from traditional lexical matching to modern neural methods.

## Methodology
We evaluated each model using Mean Squared Error (MSE) on a test set of 100 queries. Lower MSE indicates better performance.

### Implemented Models

1. **Baseline: BM25 without preprocessing**
   - Traditional lexical matching algorithm
   - MSE: 0.5282 (±0.9304)

2. **BM25 with preprocessing**
   - Enhanced BM25 with text preprocessing steps
   - MSE: 0.4885 (±0.8819)
   - Improvement: 7.52%

3. **Hybrid BM25 with Dense Retriever**
   - Combined BM25 with all-mpnet-base-v2 embeddings
   - MSE: 0.4891 (±0.8818)
   - Improvement: 7.40%

4. **ColBERT-style Retriever**
   - Late interaction model
   - MSE: 0.6222 (±0.9271)
   - Performance degradation: -17.80%

5. **BERT Bi-encoder**
   - Dense retrieval using BERT
   - MSE: 0.4546 (±0.5686)
   - Improvement: 13.93%

6. **SBERT Semantic Model**
   - Sentence-BERT based retrieval
   - MSE: 0.4838 (±0.7230)
   - Improvement: 8.41%

7. **Dual Encoder Model**
   - Best performing model
   - MSE: 0.3694 (±0.5199)
   - Improvement: 30.06%

## Results Analysis

### Performance Comparison
1. **Best Performance**: Dual Encoder Model (MSE: 0.3694)
   - 30.06% improvement over baseline
   - Lowest variance (±0.5199) among all models
2. **Worst Performance**: ColBERT-style Retriever (MSE: 0.6222)

### Best Model Architecture: Dual Encoder
The Dual Encoder model achieved superior performance through its specialized architecture:

1. **Architecture Components**:
   - Two separate BERT-based encoders:
     - Query Encoder: Optimized for short query text
     - Document Encoder: Handles longer document content
   - Shared embedding space for queries and documents
   - Similarity scoring mechanism

2. **Working Mechanism**:
   - **Encoding Phase**:
     - Queries and documents are processed independently
     - Each encoder produces dense vector representations
     - Vectors are normalized to unit length
   
   - **Matching Phase**:
     - Similarity computed using dot product between query and document vectors
     - Efficient retrieval through approximate nearest neighbor search
     - Scores normalized to match relevance labels

3. **Key Advantages**:
   - Independent processing allows for document pre-computation
   - Efficient at inference time
   - Better semantic understanding compared to lexical matching
   - Lower variance in results (±0.5199)

### Key Findings
- Simple preprocessing improved BM25 performance by 7.5%
- Neural models generally outperformed traditional lexical matching
- Dual Encoder showed the most promising results with lowest MSE and variance
- ColBERT-style retriever unexpectedly performed worse than baseline

### Observations
- Dense retrieval methods (BERT, SBERT, Dual Encoder) consistently showed lower variance in results
- Hybrid approaches (BM25 + Dense) didn't significantly improve over preprocessed BM25
- The dual encoder architecture proved most effective, possibly due to:
  - Better query-document representation learning
  - Effective handling of semantic relationships
  - Robust performance across different query types

## Conclusion
Our experiments demonstrate that while BM25 remains a strong baseline, neural approaches, particularly the Dual Encoder model, can significantly improve retrieval performance. The success of the Dual Encoder model suggests that learned representations can capture semantic relationships better than traditional lexical matching methods.


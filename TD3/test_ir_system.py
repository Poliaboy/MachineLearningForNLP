from ir_system import IRSystem
from CNN.loadCNN import loadCNN
import random

def main():
    # Load CNN dataset using the provided function
    documents, summaries = loadCNN()
    
    # Take only first 100 samples
    n_samples = 100
    documents = documents[:n_samples]
    summaries = summaries[:n_samples]
    
    # Create IR system
    ir_system = IRSystem(n_samples=n_samples)
    ir_system.load_data(documents, summaries)
    
    # Test with a few random example queries
    print("Testing IR System with random example queries...")
    random_indices = random.sample(range(n_samples), 3)  # Pick 3 random indices
    for i in random_indices:
        query = summaries[i]
        results = ir_system.retrieve(query, top_k=5)
        print("--------------------------------")
        print(f"\nQuery (Summary {i}):", query[:100], "...")
        print("\nTop 5 retrieved documents:")
        for rank, (doc_idx, score) in enumerate(results, 1):
            print(f"\nRank {rank}:")
            print(f"Document {doc_idx}: {documents[doc_idx][:100]}... (Score: {score:.3f})")
            if doc_idx == i:
                print("*** This is the correct document! ***")
                

    
    # Evaluate system performance
    mrr = ir_system.evaluate()
    print("--------------------------------")
    print(f"\nOverall System Performance:")
    print(f"Mean Reciprocal Rank: {mrr:.3f}")

if __name__ == "__main__":
    main() 
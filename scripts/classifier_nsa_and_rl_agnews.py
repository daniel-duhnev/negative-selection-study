import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import numpy as np
import pandas as pd

    
def preprocess_ag_news_data(df, n=6, category_as_self='World'):
    """Convert AG News DataFrame to sequences for NSA processing"""
    # Map category names to labels (AG News uses 0-3 for World, Sports, Business, Sci/Tech)
    category_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    
    # Extract text and convert to lowercase
    sequences = []
    labels = []
    
    for idx, row in df.iterrows():
        text = row['text'].lower()
        category = category_map[row['label']]
        
        # Generate character n-grams from text
        if len(text) >= n:
            text_ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
            sequences.extend(text_ngrams)
            # Mark as self (0) or non-self (1) based on category
            label = 0 if category == category_as_self else 1
            labels.extend([label] * len(text_ngrams))
    
    return sequences, labels

def prepare_ag_news_training_data(train_df, category_as_self='World', n=6):
    """Prepare training sequences from AG News data"""
    category_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    
    # Filter for self category only (normal data)
    self_mask = train_df['label'] == list(category_map.keys())[list(category_map.values()).index(category_as_self)]
    self_data = train_df[self_mask]
    
    train_sequences = []
    for text in self_data['text']:
        clean_text = text.lower()
        if len(clean_text) >= n:
            train_sequences.append(clean_text)
    
    return train_sequences

def prepare_ag_news_test_data(test_df, category_as_self='World', n=6):
    """Prepare test sequences and labels from AG News data"""
    category_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    
    test_sequences = []
    labels = []
    
    for idx, row in test_df.iterrows():
        text = row['text'].lower()
        category = category_map[row['label']]
        
        if len(text) >= n:
            test_sequences.append(text)
            # 0 for self (normal), 1 for anomaly
            labels.append(0 if category == category_as_self else 1)
    
    return test_sequences, labels


def hamming_distance(s1, s2):
    """Computes the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def generate_detectors(train_sequences, n=6, r=2, 
                                 num_detectors=2000, coverage_samples=100):
    """Ensures consistent detector quality through coverage monitoring"""
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    alphabet = list({c for seq in train_sequences for c in seq})
    
    detectors = set()
    coverage_history = []
    anti_clustering_threshold = 0.15
    while len(detectors) < num_detectors:
        # Generate candidate with mutation bias
        base = random.choice(train_sequences) if random.random() < 0.3 else None
        candidate = (mutate_substring(base, n, alphabet) if base 
                    else ''.join(random.choices(alphabet, k=n)))
        
        if candidate not in all_self:
            # Calculate coverage impact
            new_coverage = sum(1 for d in detectors if hamming_distance(candidate, d) <= r)
            if not detectors or (new_coverage/len(detectors)) < anti_clustering_threshold:  # Anti-clustering
                detectors.add(candidate)
                coverage_history.append(len(detectors))
                
        # Stability check - abort if 100 consecutive fails
        if len(coverage_history) > coverage_samples and \
           np.std(coverage_history[-coverage_samples:]) < 5:
            break
            
    return detectors

def mutate_substring(base, n, alphabet):
    """Creates candidates through controlled mutation."""
    if len(base) < n:
        # If base is too short for mutation with chunk size n, return a random string of length n
        return ''.join(random.choices(alphabet, k=n))
    
    # Otherwise, proceed with controlled mutation
    start = random.randint(0, len(base)-n)
    return ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                    for c in base[start:start+n]])

def classify_sequence(sequence, detectors, n, r=0):
    """Computes anomaly score based on unmatched chunks."""
    if len(sequence) < n:
        return 1.0  # Treat as anomalous if too short

    chunks = [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    unmatched_count = sum(1 for chunk in chunks if not any(hamming_distance(chunk, d) <= r for d in detectors))
    return unmatched_count / len(chunks) if chunks else 1.0


def train_rl_detector_system(train_sequences, test_sequences, labels, n=6, r=2, episodes=100):
    """
    Trains a reinforcement learning system to optimize detector generation for negative selection.
    
    Args:
        train_sequences: List of self sequences for training
        test_sequences: List of sequences to classify
        labels: Binary labels for test sequences (1 for anomaly, 0 for normal)
        n: Length of detector chunks
        r: Hamming distance threshold
        episodes: Number of training episodes
        
    Returns:
        Q-table and best detectors found
    """
    # Extract alphabet from training sequences
    alphabet = list({c for seq in train_sequences for c in seq})
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    
    # Define state features - we'll use 4 features to represent detector set properties
    state_size = 4
    # Define actions - different detector generation strategies
    action_size = 3
    
    # Initialize Q-table
    Q = np.zeros((state_size, action_size))
    
    # Learning parameters
    alpha = 0.21 # Learning rate first 0.1
    gamma = 0.9  # Discount factor
    epsilon = 0.2  # Exploration rate # first one was 0.1, second 0.2
    
    max_detectors = 2000
    print("Parameters: ",alpha,gamma,epsilon)
    # Track best detector set and its performance
    best_detectors = set()
    best_auc = 0.0
    
    def get_detector_state_features(detectors, all_self):
        """Extract meaningful state features from detector set"""
        if not detectors:
            return 0  # Initial state
        
        # Feature 1: Detector diversity (using sampling for efficiency)
        if len(detectors) > 1:
            sample_size = min(100, len(detectors))
            detector_samples = random.sample(list(detectors), sample_size)
            avg_distance = sum(hamming_distance(d1, d2) 
                            for i, d1 in enumerate(detector_samples) 
                            for d2 in detector_samples[i+1:]) / (sample_size * (sample_size - 1) / 2)
            diversity = avg_distance / n
        else:
            diversity = 0
            
        # Feature 2: Non-self coverage estimation
        sample_size = 100
        random_strings = [''.join(random.choices(alphabet, k=n)) for _ in range(sample_size)]
        non_self_strings = [s for s in random_strings if s not in all_self]
        if non_self_strings:
            coverage = sum(any(hamming_distance(s, d) <= r for d in detectors) 
                        for s in non_self_strings) / len(non_self_strings)
        else:
            coverage = 0
        
        # Feature 3: Self coverage (false positive estimation)
        sample_size = min(100, len(all_self))
        if sample_size > 0:
            self_samples = random.sample(list(all_self), sample_size)
            false_positive_rate = sum(any(hamming_distance(s, d) <= r for d in detectors)
                                    for s in self_samples) / sample_size
        else:
            false_positive_rate = 0
        
        # Feature 4: Detector distribution uniformity
        if len(detectors) > 1:
            # Calculate standard deviation of distances between detectors
            distances = [hamming_distance(d1, d2) 
                        for i, d1 in enumerate(detector_samples) 
                        for d2 in detector_samples[i+1:]]
            uniformity = 1.0 - (np.std(distances) / n if distances else 0)
        else:
            uniformity = 0
        
        # Combine features into a state vector
        state_vector = [diversity, coverage, false_positive_rate, uniformity]
        
        # Discretize state space - map to integer index
        # Each feature gets divided into 10 bins (0-9)
        discretized = [min(int(f * 10), 9) for f in state_vector]
        
        # Convert multi-dimensional state to a single index
        # Using a better encoding scheme that preserves more information
        state_index = sum(discretized[i] * (10 ** i) for i in range(len(discretized))) % state_size
        
        return state_index
    
    # Detector generation strategies
    def generate_random_detectors(num_detectors=2000):
        """Generate completely random detectors"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            candidate = ''.join(random.choices(alphabet, k=n))
            if candidate not in all_self:
                detectors.add(candidate)
                
        return detectors
    
    def generate_mutation_based_detectors(num_detectors=2000):
        """Generate detectors through mutation of self strings"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            # Select a base sequence to mutate
            base = random.choice(train_sequences)
            if len(base) < n:
                continue
                
            # Create mutated substring
            start = random.randint(0, len(base)-n)
            candidate = ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                              for c in base[start:start+n]])
            
            if candidate not in all_self:
                detectors.add(candidate)
                
        return detectors
    
    def generate_coverage_optimized_detectors(num_detectors=2000):
        """Generate detectors optimized for coverage"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate with anti-clustering strategy
            if random.random() < 0.3 and train_sequences:
                # Mutation-based
                base = random.choice(train_sequences)
                if len(base) >= n:
                    start = random.randint(0, len(base)-n)
                    candidate = ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                                      for c in base[start:start+n]])
                else:
                    candidate = ''.join(random.choices(alphabet, k=n))
            else:
                # Random generation
                candidate = ''.join(random.choices(alphabet, k=n))
            
            if candidate not in all_self:
                # Anti-clustering: check if this detector is too close to existing ones
                if not detectors or sum(1 for d in detectors if hamming_distance(candidate, d) <= r) / len(detectors) < 0.05:
                    detectors.add(candidate)
                
        return detectors
    
    # Training loop
    for episode in range(episodes):
        # Current state: features of our detector set
        state = get_detector_state_features(best_detectors, all_self)
        
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = np.argmax(Q[state])
        
        # Take action: generate detectors with different strategies
        if action == 0:
            detectors = generate_random_detectors()
        elif action == 1:
            detectors = generate_mutation_based_detectors()
        else:
            detectors = generate_coverage_optimized_detectors()
        
        # Evaluate performance
        scores = [classify_sequence(seq, detectors, n, r) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        
        # Calculate reward based on AUC improvement
        reward = auc
        
        # Update best detector set if improved
        if auc > best_auc:
            best_auc = auc
            best_detectors = detectors
            print(f"Episode {episode}: New best AUC = {best_auc:.4f}, Action = {action}")
        
        # Update Q-table
        next_state = get_detector_state_features(detectors, all_self)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Decay epsilon for less exploration over time
        epsilon = max(0.01, epsilon * 0.995)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: Current AUC = {auc:.4f}, Best AUC = {best_auc:.4f}")
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    return Q, best_detectors

if __name__ == "__main__":
    # Load AG News data
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    train_df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["test"])
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Choose which category to treat as "self" (normal)
    category_as_self = 'World' # You can change this to 'Sports', 'Business', or 'Sci/Tech'
    
    # Prepare training data (only self/normal sequences)
    train_sequences = prepare_ag_news_training_data(train_df, category_as_self)
    print(f"Prepared {len(train_sequences)} training sequences from {category_as_self} category")
    
    # Prepare test data (all categories with labels)
    test_sequences, labels = prepare_ag_news_test_data(test_df, category_as_self)
    print(f"Prepared {len(test_sequences)} test sequences")
    
    # Generate detectors using your existing function
    print("Generating detectors...")
    detectors = generate_detectors(train_sequences, n=6, r=2)
    print(f"Generated {len(detectors)} detectors")
    
    # Classify test sequences
    print("Classifying test sequences...")
    anomaly_scores = [classify_sequence(seq, detectors, n=6, r=2) for seq in test_sequences]
    
    # Calculate AUC
    auc_value = roc_auc_score(labels, anomaly_scores)
    print(f"Basic NSA AUC: {auc_value:.4f}")
    
    # Plot ROC curve for basic NSA
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Basic NSA ROC Curve (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("1-specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(f"AG News Anomaly Detection - {category_as_self} as Self")
    plt.legend()
    plt.savefig(f"ag_news_{category_as_self}_basic_nsa_roc_curve.png")
    # plt.show()
    
    # ADD THE RL TRAINING BLOCK HERE:
    print("\n" + "="*50)
    print("Starting RL-enhanced NSA training...")
    print("="*50)
    
    # Train RL system on AG News
    Q, best_detectors = train_rl_detector_system(
        train_sequences, 
        test_sequences, 
        labels, 
        n=6, 
        r=2, 
        episodes=50
    )
    
    # Evaluate RL-optimized detectors
    rl_anomaly_scores = [classify_sequence(seq, best_detectors, n, r) for seq in test_sequences]
    rl_auc_value = roc_auc_score(labels, rl_anomaly_scores)
    
    print(f"RL training completed. Best AUC: {rl_auc_value:.4f}")
    print(f"Improvement over basic NSA: {rl_auc_value - auc_value:.4f}")
    
    # Plot comparison ROC curves
    fpr_rl, tpr_rl, _ = roc_curve(labels, rl_anomaly_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"Basic NSA (AUC = {auc_value:.3f})", linewidth=2)
    plt.plot(fpr_rl, tpr_rl, label=f"RL-Enhanced NSA (AUC = {rl_auc_value:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("1-specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(f"AG News Anomaly Detection Comparison - {category_as_self} as Self")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"ag_news_{category_as_self}_comparison_roc_curve.png")
    plt.show()



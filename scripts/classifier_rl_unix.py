import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import numpy as np

def preprocess_file(file_path):
    """Reads a file and returns all sequences as a list of strings."""
    print(f"Loading sequences from {file_path}...")
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(lines)} sequences.")
    return lines

def hamming_distance(s1, s2):
    """Computes the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def mutate_substring(base, n, alphabet):
    """Creates candidates through controlled mutation."""
    if len(base) < n:
        # If base is too short for mutation with chunk size n, return a random string of length n
        candidate = ''.join(random.choices(alphabet, k=n))
        # print for debugging rare case
        # print(f"mutate_substring: base too short, generated random candidate {candidate}")
        return candidate
    
    # Otherwise, proceed with controlled mutation
    start = random.randint(0, len(base)-n)
    substring = base[start:start+n]
    # print occasionally for debugging
    # print(f"mutate_substring: mutating substring '{substring}'")
    candidate = ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                    for c in substring])
    return candidate

def classify_sequence(sequence, detectors, n, r=0):
    """Computes anomaly score based on unmatched chunks."""
    if len(sequence) < n:
        # Treat as anomalous if too short
        return 1.0

    chunks = [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    unmatched_count = sum(
        1 for chunk in chunks 
        if not any(hamming_distance(chunk, d) <= r for d in detectors)
    )
    score = unmatched_count / len(chunks) if chunks else 1.0
    return score

def train_rl_detector_system(train_sequences, test_sequences, labels, n=6, r=1, episodes=20):
    """
    Trains a reinforcement learning system to optimize detector generation for negative selection.
    """
    print("Starting RL-based detector training...")
    # Extract alphabet from training sequences
    alphabet = list({c for seq in train_sequences for c in seq})
    print(f"  Alphabet size: {len(alphabet)}")
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    print(f"  Number of unique self substrings (length {n}): {len(all_self)}")
    
    # Define state features - we'll use 4 features
    state_size = 4
    # Define actions - different detector generation strategies
    action_size = 3
    
    # Initialize Q-table
    Q = np.zeros((state_size, action_size))
    
    # Learning parameters
    alpha = 0.05  # Learning rate
    gamma = 0.9   # Discount factor
    epsilon = 0.1 # Exploration rate
    
    # Track best detector set and its performance
    best_detectors = set()
    best_auc = 0.0
    
    def get_detector_state_features(detectors, all_self):
        """Extract meaningful state features from detector set"""
        if not detectors:
            return 0  # Initial state
        
        # Feature 1: Detector diversity (sampling)
        if len(detectors) > 1:
            sample_size = min(100, len(detectors))
            detector_samples = random.sample(list(detectors), sample_size)
            avg_distance = sum(
                hamming_distance(d1, d2) 
                for i, d1 in enumerate(detector_samples) 
                for d2 in detector_samples[i+1:]
            ) / (sample_size * (sample_size - 1) / 2)
            diversity = avg_distance / n
        else:
            diversity = 0
            
        # Feature 2: Non-self coverage estimation
        sample_size = 100
        random_strings = [''.join(random.choices(alphabet, k=n)) for _ in range(sample_size)]
        non_self_strings = [s for s in random_strings if s not in all_self]
        if non_self_strings:
            coverage = sum(
                any(hamming_distance(s, d) <= r for d in detectors) 
                for s in non_self_strings
            ) / len(non_self_strings)
        else:
            coverage = 0
        
        # Feature 3: Self coverage (false positive estimation)
        sample_size = min(100, len(all_self))
        if sample_size > 0:
            self_samples = random.sample(list(all_self), sample_size)
            false_positive_rate = sum(
                any(hamming_distance(s, d) <= r for d in detectors)
                for s in self_samples
            ) / sample_size
        else:
            false_positive_rate = 0
        
        # Feature 4: Detector distribution uniformity
        if len(detectors) > 1:
            # reuse detector_samples from above for distance calc
            distances = [
                hamming_distance(d1, d2) 
                for i, d1 in enumerate(detector_samples) 
                for d2 in detector_samples[i+1:]
            ]
            uniformity = 1.0 - (np.std(distances) / n if distances else 0)
        else:
            uniformity = 0
        
        state_vector = [diversity, coverage, false_positive_rate, uniformity]
        # Discretize each feature into 10 bins
        discretized = [min(int(f * 10), 9) for f in state_vector]
        # Convert to single index
        state_index = sum(discretized[i] * (10 ** i) for i in range(len(discretized))) % state_size
        return state_index
    
    # Detector generation strategies
    def generate_random_detectors(num_detectors=1000):
        """Generate completely random detectors"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            candidate = ''.join(random.choices(alphabet, k=n))
            if candidate not in all_self:
                detectors.add(candidate)
        # print occasionally
        # print(f"  generate_random_detectors: generated {len(detectors)} detectors")
        return detectors
    
    def generate_mutation_based_detectors(num_detectors=1000):
        """Generate detectors through mutation of self strings"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            base = random.choice(train_sequences)
            if len(base) < n:
                continue
                
            start = random.randint(0, len(base)-n)
            candidate = ''.join([
                c if random.random() > 0.4 else random.choice(alphabet) 
                for c in base[start:start+n]
            ])
            
            if candidate not in all_self:
                detectors.add(candidate)
        # print occasionally
        # print(f"  generate_mutation_based_detectors: generated {len(detectors)} detectors")
        return detectors
    
    def generate_coverage_optimized_detectors(num_detectors=1000):
        """Generate detectors optimized for coverage"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            if random.random() < 0.3 and train_sequences:
                base = random.choice(train_sequences)
                if len(base) >= n:
                    start = random.randint(0, len(base)-n)
                    candidate = ''.join([
                        c if random.random() > 0.4 else random.choice(alphabet) 
                        for c in base[start:start+n]
                    ])
                else:
                    candidate = ''.join(random.choices(alphabet, k=n))
            else:
                candidate = ''.join(random.choices(alphabet, k=n))
            
            if candidate not in all_self:
                # Anti-clustering check
                if not detectors or sum(
                    1 for d in detectors if hamming_distance(candidate, d) <= r
                ) / len(detectors) < 0.15:
                    detectors.add(candidate)
        # print occasionally
        # print(f"  generate_coverage_optimized_detectors: generated {len(detectors)} detectors")
        return detectors
    
    # Training loop
    print(f"Beginning training for {episodes} episodes...")
    for episode in range(episodes):
        # Current state: features of our detector set
        state = get_detector_state_features(best_detectors, all_self)
        
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = np.argmax(Q[state])
        
        # Take action: generate detectors
        if action == 0:
            detectors = generate_random_detectors()
        elif action == 1:
            detectors = generate_mutation_based_detectors()
        else:
            detectors = generate_coverage_optimized_detectors()
        
        # Evaluate performance
        scores = [classify_sequence(seq, detectors, n, r) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        
        # Calculate reward
        reward = auc
        
        # Update best detector set if improved
        if auc > best_auc:
            best_auc = auc
            best_detectors = detectors
            print(f"Episode {episode}: New best AUC = {best_auc:.4f}, action={action}")
        
        # Update Q-table
        next_state = get_detector_state_features(detectors, all_self)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        # Print progress every 10 episodes (instead of 100 to see more feedback if episodes small)
        if episode % 10 == 0:
            print(f"Episode {episode}: Current AUC = {auc:.4f}, Best AUC = {best_auc:.4f}, epsilon = {epsilon:.3f}")
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    return Q, best_detectors

if __name__ == "__main__":
    dataset_folder = "../negative-selection-data-and-scripts/syscalls/snd-cert"  # Change to desired dataset
    print(f"Dataset folder: {dataset_folder}")
    
    # Load data
    train_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.train")
    print(f"Reading training file: {train_file}")
    train_sequences = preprocess_file(train_file)
    
    # Use first test set for training the RL system
    test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.test")
    labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.labels")
    print(f"Reading test file: {test_file} and labels: {labels_file}")
    test_sequences = preprocess_file(test_file)
    labels = [int(label) for label in preprocess_file(labels_file)]
    print(f"  Loaded {len(test_sequences)} test sequences and {len(labels)} labels.")
    
    # Train RL system
    print("Training RL detector system on test set 1...")
    Q, best_detectors = train_rl_detector_system(train_sequences, test_sequences, labels, n=6, r=1)
    
    # Evaluate on other test sets
    for i in range(2, 4):  # Test sets 2 and 3
        test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.test")
        labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.labels")
        print(f"Evaluating on test set {i}: {test_file}")
        
        test_sequences = preprocess_file(test_file)
        labels = [int(label) for label in preprocess_file(labels_file)]
        print(f"  Loaded {len(test_sequences)} sequences and {len(labels)} labels for test set {i}.")
        
        # Evaluate using best detectors found by RL
        scores = [classify_sequence(seq, best_detectors, n=6, r=1) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        print(f"Test set {i} AUC: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (RL-optimized)")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("1-specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"Receiver Operating Characteristic (AUC = {auc:.3f}) for test set {i}")
        out_path = f"{os.path.basename(dataset_folder)}_{i}_rl_optimized_roc_curve.png"
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved ROC curve to {out_path}")

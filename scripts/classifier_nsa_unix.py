import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import numpy as np

def preprocess_file(file_path):
    """Reads a file and returns all sequences as a list of strings."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def hamming_distance(s1, s2):
    """Computes the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def generate_detectors(train_sequences, n=4, r=1, 
                                 num_detectors=15000, coverage_samples=100):
    """Ensures consistent detector quality through coverage monitoring"""
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    alphabet = list({c for seq in train_sequences for c in seq})
    
    detectors = set()
    coverage_history = []
    
    while len(detectors) < num_detectors:
        # Generate candidate with mutation bias
        base = random.choice(train_sequences) if random.random() < 0.3 else None
        candidate = (mutate_substring(base, n, alphabet) if base 
                    else ''.join(random.choices(alphabet, k=n)))
        
        if candidate not in all_self:
            # Calculate coverage impact
            new_coverage = sum(1 for d in detectors if hamming_distance(candidate, d) <= r)
            if not detectors or (new_coverage/len(detectors)) < 0.15:  # Anti-clustering
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

def evaluate_dataset(folder, n = 4, r = 1):
    """Processes a dataset folder and computes AUC for each test file for multiple configurations."""
    train_file = os.path.join(folder, f"{os.path.basename(folder)}.train")
    train_sequences = preprocess_file(train_file)
    
    auc_results = {}

    print(f"Generating detectors for {os.path.basename(folder)} with n={n}, r={r}...")
    detectors = generate_detectors(train_sequences, n, r)
    print(len(detectors))
    for i in range(1, 4):
        test_file = os.path.join(folder, f"{os.path.basename(folder)}.{i}.test")
        labels_file = os.path.join(folder, f"{os.path.basename(folder)}.{i}.labels")
        
        print(f"Processing {test_file} with n={n}, r={r}...")
        
        test_sequences = preprocess_file(test_file)
        labels = [int(label) for label in preprocess_file(labels_file)]
        
        if len(test_sequences) != len(labels):
            print(f"Mismatch between test sequences and labels in {test_file}!")
            continue
        
        anomaly_scores = [classify_sequence(seq, detectors, n, r) for seq in test_sequences]
        auc_value = roc_auc_score(labels, anomaly_scores)
        auc_results[f"{os.path.basename(folder)}.{i}_n{n}_r{r}"] = auc_value
        print(auc_value)
        fpr, tpr, _ = roc_curve(labels, anomaly_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve ({os.path.basename(folder)}.{i}, n={n}, r={r})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("1-specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"Receiver Operating Characteristic (AUC = {auc_value:.3f})")
        plt.legend()
        plt.savefig(f"{os.path.basename(folder)}_{i}_n{n}_r{r}_roc_curve.png")
        plt.close()
    
    return auc_results

def optimize_parameters(folder, n_range=range(3,8), r_range=range(0, 3)):
    """Grid search for optimal parameters"""
    best_params = {}
    train_file = os.path.join(folder, f"{os.path.basename(folder)}.train")
    train_sequences = preprocess_file(train_file)
    
    for n in n_range:
        detectors = generate_detectors(train_sequences, n)
        print(len(detectors))
        for r in r_range:
            # Cross-validate using first test set
            test_file = os.path.join(folder, f"{os.path.basename(folder)}.1.test")
            labels_file = os.path.join(folder, f"{os.path.basename(folder)}.1.labels")
            
            test_sequences = preprocess_file(test_file)
            labels = [int(label) for label in preprocess_file(labels_file)]
            
            scores = [classify_sequence(seq, detectors, n, r) for seq in test_sequences]
            auc_value = roc_auc_score(labels, scores)
            print(n,r, "AUC", auc_value)
            if auc_value > best_params.get('auc', 0):
                best_params = {'n': n, 'r': r, 'auc': auc_value}    
    return best_params

def evaluate_and_plot_dataset(folder, n=4, r=1):
    """
    Processes a dataset folder, computes overall AUC, individual AUCs,
    and generates plots (overall ROC curve and box plot for detector counts).
    """
    train_file = os.path.join(folder, f"{os.path.basename(folder)}.train")
    train_sequences = preprocess_file(train_file)
    
    print(f"Generating detectors for {os.path.basename(folder)} with n={n}, r={r}...")
    detectors = generate_detectors(train_sequences, n, r)
    print(f"Generated {len(detectors)} detectors.")
    
    all_labels = []
    all_scores = []
    auc_results = []
    
    for i in range(1, 4):  # Assuming there are 3 test files per dataset
        test_file = os.path.join(folder, f"{os.path.basename(folder)}.{i}.test")
        labels_file = os.path.join(folder, f"{os.path.basename(folder)}.{i}.labels")
        
        print(f"Processing {test_file} with n={n}, r={r}...")
        
        test_sequences = preprocess_file(test_file)
        labels = [int(label) for label in preprocess_file(labels_file)]
        
        if len(test_sequences) != len(labels):
            print(f"Mismatch between test sequences and labels in {test_file}!")
            continue
        
        # Compute anomaly scores
        anomaly_scores = [classify_sequence(seq, detectors, n, r) for seq in test_sequences]
        
        # Compute AUC for this file
        auc_value = roc_auc_score(labels, anomaly_scores)
        auc_results.append((f"{os.path.basename(folder)}.{i}", auc_value))
        print(f"AUC for {test_file}: {auc_value}")
        
        # Append scores and labels for overall calculation
        all_scores.extend(anomaly_scores)
        all_labels.extend(labels)
    
    # Compute overall AUC
    if all_labels and all_scores:
        overall_auc = roc_auc_score(all_labels, all_scores)
        print(f"Overall AUC for {os.path.basename(folder)}: {overall_auc}")
        
        # Plot overall ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=overall_auc).plot()
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("1-specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"Overall ROC Curve for {os.path.basename(folder)} (AUC = {overall_auc:.3f})")
        plt.savefig(f"{os.path.basename(folder)}_overall_roc_curve_n{n}_r{r}.png")
        plt.close()
    
    # Generate box plot for individual AUCs
    if auc_results:
        auc_values = [auc[1] for auc in auc_results]
        labels = [auc[0] for auc in auc_results]
        
        plt.figure()
        plt.boxplot(auc_values, vert=True, patch_artist=True)
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
        plt.ylabel("AUC Value")
        plt.title(f"AUC Values Across Test Files ({os.path.basename(folder)})")
        plt.savefig(f"{os.path.basename(folder)}_auc_box_plot.png")
        plt.close()
    
    return overall_auc, auc_results

if __name__ == "__main__":
    dataset_folder_cert = "syscalls/snd-cert"
    dataset_folder_unm = "syscalls/snd-unm"
    
    # Evaluate snd-cert dataset
    overall_auc_cert, auc_scores_cert = evaluate_and_plot_dataset(dataset_folder_cert)
    print("AUC Scores (snd-cert):", auc_scores_cert)
    print("Overall AUC (snd-cert):", overall_auc_cert)

    # Evaluate snd-unm dataset
    overall_auc_unm, auc_scores_unm = evaluate_and_plot_dataset(dataset_folder_unm)
    print("AUC Scores (snd-unm):", auc_scores_unm)
    print("Overall AUC (snd-unm):", overall_auc_unm)


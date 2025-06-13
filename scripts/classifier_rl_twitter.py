import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import html
from datetime import datetime
import json
import time
from collections import defaultdict
import seaborn as sns

def create_output_dir(base="results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{now}")
    os.makedirs(path, exist_ok=True)
    return path

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Replace URLs
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    # Replace mentions
    text = re.sub(r'@\w+', ' <USER> ', text)
    # Normalize hashtags
    text = re.sub(r'#(\w+)', r' <HASHTAG_\1> ', text)
    # Replace non-ASCII (e.g. emojis) with placeholder
    def replace_non_ascii(match):
        return ' <EMOJI> '
    text = re.sub(r'[^\x00-\x7F]+', replace_non_ascii, text)
    # Cleanup HTML leftovers
    text = text.replace('\xa0', ' ')
    # Collapse repeated punctuation: "!!!" -> "!!"
    text = re.sub(r'([!?.]){2,}', r'\1\1', text)
    # Collapse repeated chars: "soooo" -> "soo"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Remove unwanted symbols, keep alphanumerics, placeholders, spaces
    text = re.sub(r'[^A-Za-z0-9<>\s_]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    return text

def sample_sentiment140(path, n_pos=10000, n_neg=10000, chunksize=200_000, random_state=42):
    """
    Read CSV in chunks, sample up to n_pos positives and n_neg negatives.
    Returns a DataFrame with exactly n_pos rows of sentiment=4 and n_neg rows of sentiment=0 (if enough exist).
    """
    pos_list = []
    neg_list = []
    need_pos = n_pos
    need_neg = n_neg
    rng = random.Random(random_state)

    cols = ['sentiment','text']
    # We only need sentiment and text
    for chunk in pd.read_csv(
        path,
        sep=',',
        header=None,
        names=cols,
        usecols=['sentiment','text'],
        encoding='latin-1',
        engine='python',
        on_bad_lines='skip',
        chunksize=chunksize
    ):
        # Ensure sentiment is numeric
        chunk['sentiment'] = pd.to_numeric(chunk['sentiment'], errors='coerce')
        # Filter to only 0 or 4
        chunk = chunk[chunk['sentiment'].isin([0,4])]
        if chunk.empty:
            continue

        # Separate positives and negatives
        pos_chunk = chunk[chunk['sentiment'] == 4]
        neg_chunk = chunk[chunk['sentiment'] == 0]

        # Sample from pos_chunk up to need_pos
        if need_pos > 0 and not pos_chunk.empty:
            if len(pos_chunk) <= need_pos:
                sampled_pos = pos_chunk
            else:
                sampled_pos = pos_chunk.sample(n=need_pos, random_state=rng.randint(0, 10**9))
            pos_list.append(sampled_pos)
            need_pos -= len(sampled_pos)

        # Sample from neg_chunk up to need_neg
        if need_neg > 0 and not neg_chunk.empty:
            if len(neg_chunk) <= need_neg:
                sampled_neg = neg_chunk
            else:
                sampled_neg = neg_chunk.sample(n=need_neg, random_state=rng.randint(0, 10**9))
            neg_list.append(sampled_neg)
            need_neg -= len(sampled_neg)

        # If both needs met, break
        if need_pos <= 0 and need_neg <= 0:
            break

    # Concatenate results
    if pos_list:
        df_pos = pd.concat(pos_list, ignore_index=True)
    else:
        df_pos = pd.DataFrame(columns=['sentiment','text'])
    if neg_list:
        df_neg = pd.concat(neg_list, ignore_index=True)
    else:
        df_neg = pd.DataFrame(columns=['sentiment','text'])

    # If we collected more than needed (possible if chunk sizes cause slight over-collection), sample down:
    if len(df_pos) > n_pos:
        df_pos = df_pos.sample(n=n_pos, random_state=random_state).reset_index(drop=True)
    if len(df_neg) > n_neg:
        df_neg = df_neg.sample(n=n_neg, random_state=random_state).reset_index(drop=True)

    # Combine and shuffle
    df_small = pd.concat([df_pos, df_neg], ignore_index=True)
    df_small = df_small.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add clean_text
    df_small['clean_text'] = df_small['text'].apply(clean_tweet)
    # Ensure sentiment is int
    df_small['sentiment'] = df_small['sentiment'].astype(int)
    return df_small

def prepare_sentiment140_data(df, self_label=4, n=9, train_frac=0.8, random_state=42):
    """
    Splits into train and test where:
      - train_sequences: only text where sentiment is self
      - test_sequences: all texts - self and non-self
      - labels: 0 for self sentiment, 1 for non-self sentiment
    Returns: train_sequences, test_sequences, labels
    """
    # shuffle & split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # only self texts for detector training
    train_sequences = [
        txt
        for txt in train_df.loc[train_df.sentiment == self_label, 'clean_text']
        if isinstance(txt, str) and len(txt) >= n
    ]

    # full test set with binary labels
    test_sequences = []
    labels = []
    for txt, sent in zip(test_df['clean_text'], test_df['sentiment']):
        if not isinstance(txt, str):
            # skip null / non-text rows
            continue
        t = txt
        if len(t) >= n:
            test_sequences.append(t)
            labels.append(0 if sent == self_label else 1)

    return train_sequences, test_sequences, labels

def hamming_distance(s1, s2):
    """Computes the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def classify_sequence(sequence, detectors, n, r=3):
    """Computes anomaly score based on unmatched chunks."""
    if len(sequence) < n:
        return 1.0  # Treat as anomalous if too short

    chunks = [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    unmatched_count = sum(1 for chunk in chunks if not any(hamming_distance(chunk, d) <= r for d in detectors))
    return unmatched_count / len(chunks) if chunks else 1.0

def train_rl_detector_system(train_sequences, test_sequences, labels, n, r, episodes=20):
    """
    RL loop to optimize NSA detector generation strategies for given n, r.
    Returns: Q-table (dict) and best_detectors set.
    """
    t0 = time.time()
    # Build alphabet and all_self substrings from train_sequences
    alphabet = list({c for seq in train_sequences for c in seq})
    # Size info
    print(f"Building all_self substrings (n={n}) from {len(train_sequences)} training sequences...")
    t_build = time.time()
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    print(f"  all_self size: {len(all_self):,} substrings built in {time.time()-t_build:.2f}s; total init time {time.time()-t0:.2f}s")

    action_size = 3
    Q = defaultdict(lambda: np.zeros(action_size))
    alpha = 0.05
    gamma = 0.9
    epsilon = 0.1

    best_detectors = set()
    best_auc = 0.0

    def get_detector_state_features(detectors):
        """Return a tuple of discretized features (diversity, coverage, fp_rate, uniformity)."""
        if not detectors:
            return (0,0,0,0)
        # Diversity
        if len(detectors) > 1:
            sample_size = min(50, len(detectors))
            samples = random.sample(list(detectors), sample_size)
            total = 0; count = 0
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    total += hamming_distance(samples[i], samples[j])
                    count += 1
            avg_distance = total / count if count>0 else 0
            diversity = avg_distance / n
        else:
            diversity = 0.0
            samples = []
        # Non-self coverage estimate
        rand_samples = [''.join(random.choices(alphabet, k=n)) for _ in range(50)]
        non_self = [s for s in rand_samples if s not in all_self]
        if non_self:
            coverage = sum(any(hamming_distance(s, d) <= r for d in detectors) for s in non_self) / len(non_self)
        else:
            coverage = 0.0
        # False-positive rate on self samples
        if all_self:
            self_samples = random.sample(list(all_self), min(50, len(all_self)))
            fp = sum(any(hamming_distance(s, d) <= r for d in detectors) for s in self_samples) / len(self_samples)
        else:
            fp = 0.0
        # Uniformity
        if len(detectors) > 1 and samples:
            distances = []
            for i in range(len(samples)):
                for j in range(i+1, len(samples)):
                    distances.append(hamming_distance(samples[i], samples[j]))
            uniformity = 1.0 - (np.std(distances)/n if distances else 0.0)
            uniformity = max(0.0, min(uniformity, 1.0))
        else:
            uniformity = 0.0
        def disc(x):
            idx = int(x*10)
            return max(0, min(idx, 9))
        return (disc(diversity), disc(coverage), disc(fp), disc(uniformity))

    # Detector generation strategies
    def generate_random_detectors(num_detectors=2000):
        detectors = set()
        attempts = 0
        while len(detectors) < num_detectors and attempts < num_detectors*10:
            attempts += 1
            cand = ''.join(random.choices(alphabet, k=n))
            if cand not in all_self:
                detectors.add(cand)
        return detectors

    def generate_mutation_based_detectors(num_detectors=2000):
        detectors = set()
        attempts = 0
        while len(detectors) < num_detectors and attempts < num_detectors*10:
            attempts += 1
            base = random.choice(train_sequences)
            if len(base) < n:
                continue
            start = random.randint(0, len(base)-n)
            cand = ''.join([c if random.random()>0.4 else random.choice(alphabet)
                            for c in base[start:start+n]])
            if cand not in all_self:
                detectors.add(cand)
        return detectors

    def generate_coverage_optimized_detectors(num_detectors=2000):
        detectors = set()
        attempts = 0
        while len(detectors) < num_detectors and attempts < num_detectors*10:
            attempts += 1
            if random.random() < 0.3 and train_sequences:
                base = random.choice(train_sequences)
                if len(base) >= n:
                    start = random.randint(0, len(base)-n)
                    cand = ''.join([c if random.random()>0.4 else random.choice(alphabet)
                                    for c in base[start:start+n]])
                else:
                    cand = ''.join(random.choices(alphabet, k=n))
            else:
                cand = ''.join(random.choices(alphabet, k=n))
            if cand not in all_self:
                if not detectors or sum(1 for d in detectors if hamming_distance(cand,d) <= r)/len(detectors) < 0.05:
                    detectors.add(cand)
        return detectors

    print(f"Starting RL for {episodes} episodes on test size {len(test_sequences)}")
    for episode in range(episodes):
        # get state from current best_detectors
        state = get_detector_state_features(best_detectors)
        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, action_size-1)
        else:
            action = int(np.argmax(Q[state]))
        # generate new candidate detectors set
        if action == 0:
            dets = generate_random_detectors()
        elif action == 1:
            dets = generate_mutation_based_detectors()
        else:
            dets = generate_coverage_optimized_detectors()
        # evaluate
        scores = [classify_sequence(seq, dets, n, r) for seq in test_sequences]
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0
        # if improved then update best
        if auc > best_auc:
            best_auc = auc
            best_detectors = dets
            print(f" Episode {episode}: new best AUC={best_auc:.4f}, action={action}")
        # Q-update
        next_state = get_detector_state_features(dets)
        Q[state][action] += alpha * (auc + gamma * np.max(Q[next_state]) - Q[state][action])
        # decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
    total_time = time.time() - t0
    print(f"RL done. Best AUC={best_auc:.4f}, total time {total_time:.1f}s")
    return Q, best_detectors

if __name__ == "__main__":
    # Load the CSV
    df = sample_sentiment140("../data/twitter-sentiment/twitter-text-sentiment-data.csv")
    print(f"Loaded {len(df)} tweets. Positives: {(df.sentiment==4).sum()}, Negatives: {(df.sentiment==0).sum()}")

    # Split the dataset into train and test
    train_raw, test_raw = train_test_split(
        df, 
        test_size=0.2,     
        stratify=df['sentiment'], 
        random_state=42
    )

    n = 9
    r = 3
    self_label = 4 

    # Build positive training sequences
    train_seqs = [
        txt for txt in train_raw.loc[train_raw.sentiment == self_label, 'clean_text']
        if isinstance(txt, str) and len(txt) >= n
    ]

    # Build test sequences and labels
    test_seqs = []
    labels     = []
    for txt, sent in zip(test_raw['clean_text'], test_raw['sentiment']):
        if not isinstance(txt, str): 
            continue
        t = txt
        if len(t) < n: 
            continue
        test_seqs.append(t)
        # 0 for self (positive) and 1 for anomaly (negative)
        labels.append(0 if sent == self_label else 1)

    print(f"Training on {len(train_seqs)} positives from train split.")
    print(f"Testing on {len(test_seqs)} tweets [{labels.count(0)} normals, {labels.count(1)} anomalies].")

    output_dir = create_output_dir()

    print(f"Running RL-enhanced NSA with n={n}, r={r} ...")
    Q, best_detectors = train_rl_detector_system(train_seqs, test_seqs, labels, n=n, r=r, episodes=50)

    # Compute RL AUC
    rl_scores = [classify_sequence(s, best_detectors, n, r) for s in test_seqs]
    rl_auc = roc_auc_score(labels, rl_scores)
    print(f"RL-enhanced NSA AUC (n={n}, r={r}): {rl_auc:.4f}")

    # Save RL metrics
    with open(os.path.join(output_dir, "rl_metrics.json"), "w") as f:
        json.dump({"n": n, "r": r, "rl_auc": rl_auc}, f, indent=2)

    # Save best detectors
    with open(os.path.join(output_dir, "rl_detectors.txt"), "w") as f:
        for d in best_detectors:
            f.write(d + "\n")

    # Plot only RL ROC curve
    fpr_rl, tpr_rl, _ = roc_curve(labels, rl_scores)
    plt.figure()
    plt.plot(fpr_rl, tpr_rl, label=f"RL-NSA (AUC={rl_auc:.3f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rl_roc_curve.png"))
    plt.close()

    print(f"RL results saved under {output_dir}")


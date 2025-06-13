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
import seaborn as sns

def create_output_dir(base="results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{now}")
    os.makedirs(path, exist_ok=True)
    return path

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    # 1) Decode HTML entities
    text = html.unescape(text)
    # 2) Replace URLs
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    # 3) Replace mentions
    text = re.sub(r'@\w+', ' <USER> ', text)
    # 4) Normalize hashtags
    text = re.sub(r'#(\w+)', r' <HASHTAG_\1> ', text)
    # 5) Replace non-ASCII (e.g. emojis) with placeholder
    def replace_non_ascii(match):
        return ' <EMOJI> '
    text = re.sub(r'[^\x00-\x7F]+', replace_non_ascii, text)
    # 6) Cleanup HTML leftovers
    text = text.replace('\xa0', ' ')
    # 7) Collapse repeated punctuation: "!!!" -> "!!"
    text = re.sub(r'([!?.]){2,}', r'\1\1', text)
    # 8) Collapse repeated chars: "soooo" -> "soo"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 9) Remove unwanted symbols, keep alphanumerics, placeholders, spaces
    text = re.sub(r'[^A-Za-z0-9<>\s_]', ' ', text)
    # 10) Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 11) Lowercase
    text = text.lower()
    return text

def sample_sentiment140(path, n_pos=1000, n_neg=1000, chunksize=200_000, random_state=42):
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

def prepare_sentiment140_data(df, self_label=4, n=6, train_frac=0.8, random_state=42):
    """
    Splits into train/test, then:
      - train_sequences: only texts whose sentiment == self_label (normal)
      - test_sequences: all texts
      - labels:        0 if sentiment == self_label else 1
    Returns: train_sequences, test_sequences, labels
    """
    # shuffle & split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # only “self” texts for detector training
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

def generate_detectors(train_sequences, n=6, r=2, 
                                 num_detectors=10000, coverage_samples=100):
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

def grid_search_n_r(train_texts, test_pairs, self_label, n_values, r_values,
                    coverage_samples=100, output_dir=None):
    """
    train_texts: list of raw text strings for positive class
    test_pairs: list of (text, sentiment) tuples for test data
    self_label: label value considered 'self' (e.g., 4 for positive tweets)
    n_values: iterable of candidate n
    r_values: iterable of candidate r
    coverage_samples: passed to generate_detectors
    output_dir: directory where to save the heatmap
    """
    results = []
    best_auc = -1
    best_params = {}

    for n in n_values:
        print(f"  Evaluating n={n}...")
        # Filter training sequences for this n
        train_seqs = [txt for txt in train_texts if isinstance(txt, str) and len(txt) >= n]
        # Build test sequences and labels for this n
        test_seqs = []
        labels = []
        for txt, sent in test_pairs:
            if not isinstance(txt, str):
                continue
            if len(txt) < n:
                continue
            test_seqs.append(txt)
            labels.append(0 if sent == self_label else 1)
        print(f"(Grid) n={n}: {len(train_seqs)} train sequences, {len(test_seqs)} test sequences")

        for r in r_values:
            print(f"  Evaluating r={r}...")
            detectors = generate_detectors(train_seqs, n=n, r=r, coverage_samples=coverage_samples)
            scores = [classify_sequence(s, detectors, n=n, r=r) for s in test_seqs]
            # If no test_seqs, skip
            if len(test_seqs) == 0:
                print(f"    Skipping n={n}, r={r} because no test sequences of length ≥ {n}")
                continue
            auc = roc_auc_score(labels, scores)
            print(f"    AUC for n={n}, r={r}: {auc:.4f}")

            results.append((n, r, auc))
            if auc > best_auc:
                best_auc = auc
                best_params = {'n': n, 'r': r, 'auc': auc}

    # Build DataFrame and pivot for heatmap
    df_auc = pd.DataFrame(results, columns=['n', 'r', 'auc'])
    if not df_auc.empty:
        auc_pivot = df_auc.pivot(index='n', columns='r', values='auc')

        # Plot heatmap if output_dir given
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            sns.heatmap(auc_pivot, annot=True, fmt=".3f", cmap="viridis")
            plt.title("AUC for different n and r values")
            plt.xlabel("r")
            plt.ylabel("n")
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, "grid_search_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            print(f"Saved heatmap to {heatmap_path}")
    else:
        print("No valid results to plot heatmap.")

    return best_params, results


if __name__ == "__main__":
    # 1) Load the CSV
    df = sample_sentiment140("twitter-text-sentiment-data.csv")
    print(f"Loaded {len(df)} tweets. Positives: {(df.sentiment==4).sum()}, Negatives: {(df.sentiment==0).sum()}")

    # 2) Split the dataset into train and test
    train_raw, test_raw = train_test_split(
        df,
        test_size=0.2,
        stratify=df['sentiment'],
        random_state=42
    )

    self_label = 4  # positive class label

    # Prepare raw train_texts (only positives) and test_pairs
    train_texts = [
        txt for txt in train_raw.loc[train_raw.sentiment == self_label, 'clean_text']
        if isinstance(txt, str)
    ]
    test_pairs = [
        (txt, sent)
        for txt, sent in zip(test_raw['clean_text'], test_raw['sentiment'])
        if isinstance(txt, str)
    ]

    print(f"Available positive training texts (unfiltered by n): {len(train_texts)}")
    print(f"Available test texts (unfiltered by n): {len(test_pairs)}")

    # 3) Prepare output directory early
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")

    # 4) Grid search for n and r ranges
    n_values = range(6, 10)
    r_values = range(1, 3)
    print(f"Starting grid search over n={list(n_values)} and r={list(r_values)}")
    best_params, all_results = grid_search_n_r(
        train_texts, test_pairs, self_label,
        n_values, r_values,
        coverage_samples=100,
        output_dir=output_dir
    )
    print("Grid search complete. Best parameters:", best_params)

    # Save full grid results to CSV
    df_grid = pd.DataFrame(all_results, columns=['n', 'r', 'auc'])
    df_grid.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)

    # Save best_params
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # 5) Final run with best n, r
    if best_params:
        n_best = best_params['n']
        r_best = best_params['r']
        print(f"Using best n={n_best}, r={r_best} for final NSA run")

        # Filter sequences again for best n
        train_seqs_final = [txt for txt in train_texts if len(txt) >= n_best]
        test_seqs_final = []
        labels_final = []
        for txt, sent in test_pairs:
            if len(txt) < n_best:
                continue
            test_seqs_final.append(txt)
            labels_final.append(0 if sent == self_label else 1)

        print(f"Final: {len(train_seqs_final)} train sequences, {len(test_seqs_final)} test sequences")

        # Generate detectors and evaluate
        detectors = generate_detectors(train_seqs_final, n=n_best, r=r_best)
        print(f"Generated {len(detectors)} detectors with n={n_best}, r={r_best}.")
        scores = [classify_sequence(s, detectors, n=n_best, r=r_best) for s in test_seqs_final]
        if len(test_seqs_final) > 0:
            auc_final = roc_auc_score(labels_final, scores)
            print(f"Plain NSA AUC (n={n_best}, r={r_best}): {auc_final:.4f}")
        else:
            auc_final = None
            print("No test sequences for final n; cannot compute AUC.")

        # Save final metrics
        metrics = {
            "n": n_best,
            "r": r_best,
            "auc": auc_final,
            "train_size": len(train_seqs_final),
            "test_size": len(test_seqs_final),
            "normal_count": labels_final.count(0) if labels_final else 0,
            "anomaly_count": labels_final.count(1) if labels_final else 0
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save scores and labels
        if test_seqs_final:
            pd.DataFrame({
                "sequence": test_seqs_final,
                "label": labels_final,
                "score": scores
            }).to_csv(os.path.join(output_dir, "anomaly_scores.csv"), index=False)

            # Plot ROC curve and save
            fpr, tpr, _ = roc_curve(labels_final, scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_final:.3f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("1 - Specificity (FPR)")
            plt.ylabel("Sensitivity (TPR)")
            plt.title(f"NSA ROC Curve (n={n_best}, r={r_best})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roc_curve.png"))
            plt.close()

        # Save detectors
        with open(os.path.join(output_dir, "detectors.txt"), "w") as f:
            for d in detectors:
                f.write(d + "\n")

        print(f"Final results saved to {output_dir}")
    else:
        print("No best parameters found from grid search.")
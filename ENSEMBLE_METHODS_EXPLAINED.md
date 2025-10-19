# Ensemble Methods Explained

## Overview

Ensemble methods in this fraud detection system combine multiple detection algorithms to create more robust and accurate fraud rankings. Think of it as a "panel of experts" where each expert (detection method) contributes their opinion, and the final decision is based on their collective wisdom.

## Core Concept

**Problem**: Different fraud detection methods are good at detecting different patterns:
- **PageRank** finds accounts central in the transaction network
- **In-Degree** finds accounts receiving many transactions (collectors)
- **Transaction Count** finds highly active accounts
- **K-Core** finds tightly connected groups

**Solution**: Combine them! An account flagged by multiple methods is more likely to be fraudulent.

---

## How Ensemble Scores Are Calculated

### Step 1: Score Normalization

Each method produces raw scores on different scales:
- In-Degree might range from 1 to 10,000
- PageRank might range from 0.0001 to 0.05
- Transaction count might range from 5 to 50,000

**Normalization Formula** (Min-Max Scaling):
```
normalized_score = (raw_score - min_score) / (max_score - min_score)
```

**Result**: All scores are now on a [0, 1] scale where:
- 0 = lowest score in that method
- 1 = highest score in that method
- 0.5 = middle score

### Step 2: Weighted Combination

Each method gets a weight based on its expected performance:

```python
ensemble_score(node) = Σ (weight_i × normalized_score_i(node))
```

**Example** (ensemble_top3):
```
ensemble_score = 0.40 × in_deg_norm 
               + 0.35 × pagerank_norm 
               + 0.25 × in_tx_norm
```

### Step 3: Missing Value Handling

If a node doesn't have a score in a particular method (rare), it gets the default value:
- **0.0** if the method has varied scores (was successfully normalized)
- **0.5** if all scores in that method are identical (neutral)

---

## Ensemble Methods Available

### 1. **ensemble_top3** ⭐ Most Common

**Purpose**: Combines the three best-performing individual methods

**Components**:
- `in_deg` (40%) - In-degree centrality
- `pagerank_wlog` (35%) - PageRank weighted by log-transaction amounts
- `in_tx` (25%) - Incoming transaction count

**When to Use**: General-purpose fraud detection, proven combination

**Why These Weights**:
- In-degree gets highest weight (40%) because it's consistently good
- PageRank gets 35% as it captures network centrality well
- Transaction count gets 25% to avoid over-weighting volume

**Code**:
```python
ensembles['ensemble_top3'] = ensemble_scores([
    score_dict['in_deg'],
    score_dict['pagerank_wlog'],
    score_dict['in_tx']
], weights=[0.40, 0.35, 0.25])
```

---

### 2. **ensemble_diverse**

**Purpose**: Broader coverage using four complementary methods

**Components**:
- `in_deg` (30%) - In-degree centrality
- `pagerank_wlog` (30%) - PageRank weighted
- `kcore_in` (20%) - K-core decomposition (incoming)
- `in_tx` (20%) - Incoming transaction count

**When to Use**: When you want to capture different fraud patterns

**Why Diverse**:
- In-degree and PageRank (centrality methods)
- K-core (cohesive subgraph method)
- Transaction count (volume method)
- Together they cover: centrality, structure, and activity

**Code**:
```python
ensembles['ensemble_diverse'] = ensemble_scores([
    score_dict['in_deg'],
    score_dict['pagerank_wlog'],
    score_dict['kcore_in'],
    score_dict['in_tx']
], weights=[0.30, 0.30, 0.20, 0.20])
```

---

### 3. **ensemble_pattern**

**Purpose**: Incorporates heuristic pattern features

**Components**:
- `in_deg` (40%)
- `pagerank_wlog` (35%)
- `pattern_features` (25%) - Heuristic fraud patterns

**When to Use**: When pattern-based detection is available

**Note**: In practice, pattern_features often performed poorly (see your metrics), so this ensemble may underperform ensemble_top3

**Code**:
```python
ensembles['ensemble_pattern'] = ensemble_scores([
    score_dict['in_deg'],
    score_dict['pagerank_wlog'],
    score_dict['pattern_features']
], weights=[0.40, 0.35, 0.25])
```

---

### 4. **ensemble_ultimate**

**Purpose**: "Kitchen sink" approach - combines everything available

**Components** (if all available):
- `in_deg` (25%)
- `pagerank_wlog` (25%)
- `in_tx` (20%)
- `kcore_in` (15%)
- `pattern_features` (15%) - if available

**Fallback** (without pattern_features):
- `in_deg` (30%)
- `pagerank_wlog` (30%)
- `in_tx` (20%)
- `kcore_in` (20%)

**When to Use**: Maximum coverage, but may be diluted by poor methods

**Code**:
```python
ultimate_keys = ['in_deg', 'pagerank_wlog', 'in_tx', 'kcore_in']
if 'pattern_features' in score_dict:
    ultimate_keys.append('pattern_features')
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
else:
    weights = [0.30, 0.30, 0.20, 0.20]
ensembles['ensemble_ultimate'] = ensemble_scores(ultimate_keys, weights)
```

---

### 5. **ensemble_seeded**

**Purpose**: Incorporates semi-supervised learning via seeded PageRank

**Components** (with pattern_features):
- `in_deg` (35%)
- `pagerank_wlog` (30%)
- `seeded_pr` (20%) - PageRank seeded with known fraud
- `pattern_features` (15%)

**Components** (without pattern_features):
- `in_deg` (45%)
- `pagerank_wlog` (35%)
- `seeded_pr` (20%)

**When to Use**: When you have labeled fraud examples to seed with

**Special Feature**: Uses historical fraud labels to bias detection

**Code**:
```python
if 'seeded_pr' in score_dict and score_dict.get('seeded_pr'):
    seeded_keys = ['in_deg', 'pagerank_wlog', 'seeded_pr']
    if 'pattern_features' in score_dict:
        seeded_keys.append('pattern_features')
        weights = [0.35, 0.30, 0.20, 0.15]
    else:
        weights = [0.45, 0.35, 0.20]
    ensembles['ensemble_seeded'] = ensemble_scores(seeded_keys, weights)
```

---

## Complete Algorithm Walkthrough

### Example Calculation

**Input**: 3 accounts with raw scores from 3 methods

| Account | In-Degree (raw) | PageRank (raw) | In-Tx (raw) |
|---------|----------------|----------------|-------------|
| A       | 100            | 0.05           | 500         |
| B       | 50             | 0.02           | 300         |
| C       | 200            | 0.10           | 800         |

**Step 1: Normalize Each Method**

**In-Degree**:
- Min = 50, Max = 200
- A: (100 - 50) / (200 - 50) = 0.33
- B: (50 - 50) / (200 - 50) = 0.00
- C: (200 - 50) / (200 - 50) = 1.00

**PageRank**:
- Min = 0.02, Max = 0.10
- A: (0.05 - 0.02) / (0.10 - 0.02) = 0.375
- B: (0.02 - 0.02) / (0.10 - 0.02) = 0.00
- C: (0.10 - 0.02) / (0.10 - 0.02) = 1.00

**In-Tx**:
- Min = 300, Max = 800
- A: (500 - 300) / (800 - 300) = 0.40
- B: (300 - 300) / (800 - 300) = 0.00
- C: (800 - 300) / (800 - 300) = 1.00

**Step 2: Apply Weights** (ensemble_top3: 0.40, 0.35, 0.25)

**Account A**:
```
ensemble_A = 0.40 × 0.33 + 0.35 × 0.375 + 0.25 × 0.40
          = 0.132 + 0.131 + 0.10
          = 0.363
```

**Account B**:
```
ensemble_B = 0.40 × 0.00 + 0.35 × 0.00 + 0.25 × 0.00
          = 0.00
```

**Account C**:
```
ensemble_C = 0.40 × 1.00 + 0.35 × 1.00 + 0.25 × 1.00
          = 1.00
```

**Final Ranking**: C (1.00) > A (0.363) > B (0.00)

---

## Evaluation Metrics for Ensembles

Once ensemble scores are computed, they're evaluated the same way as individual methods:

### 1. **Average Precision (AP)**
Measures ranking quality across all thresholds

### 2. **Precision at k%**
- `p_at_0.5pct`: Precision in top 0.5% of ranked accounts
- `p_at_1.0pct`: Precision in top 1%
- `p_at_2.0pct`: Precision in top 2%
- `p_at_5.0pct`: Precision in top 5%

### 3. **Attempt Coverage**
- `attcov_at_X%`: What % of fraud schemes are detected in top X%

### 4. **Lift**
- `lift_p_at_1.0pct`: How much better than random at 1% threshold
- Random baseline precision = prevalence (e.g., 1.25%)
- Lift = (Actual precision) / (Random precision)

---

## Why Ensembles Usually Work Better

### 1. **Error Averaging**
Individual methods make mistakes, but different mistakes. Combining reduces overall error.

### 2. **Complementary Strengths**
- PageRank finds network hubs
- In-degree finds transaction collectors
- K-core finds tightly connected groups
- Together: comprehensive coverage

### 3. **Robustness**
If one method fails or performs poorly, others compensate.

### 4. **Reduced Variance**
Ensemble predictions are more stable than individual methods.

---

## Performance Comparison (From Your Data)

Based on typical results:

| Method | Median AP | Median P@1% | Interpretation |
|--------|-----------|-------------|----------------|
| `ensemble_top3` | ~0.0255 | ~6.8% | Best overall ensemble |
| `ensemble_diverse` | ~0.0248 | ~6.5% | Good, broader coverage |
| `ensemble_ultimate` | ~0.0235 | ~6.2% | Diluted by pattern_features |
| `ensemble_pattern` | ~0.0210 | ~5.5% | Hurt by poor pattern_features |
| `ensemble_seeded` | ~0.0268 | ~7.1% | Best when seeds available |
| `pagerank_wlog` | ~0.0272 | ~7.3% | Best individual |
| `in_deg` | ~0.0268 | ~7.3% | Second best individual |
| `random` | ~0.0125 | ~1.25% | Baseline |

**Key Insight**: Ensembles are competitive with best individuals, more robust.

---

## When to Use Which Ensemble

### Use **ensemble_top3** when:
✅ You want a reliable, proven combination  
✅ General-purpose fraud detection  
✅ You don't have labeled seeds  
✅ You want simplicity and interpretability

### Use **ensemble_diverse** when:
✅ You want maximum pattern coverage  
✅ Different fraud types are expected  
✅ You want to hedge your bets

### Use **ensemble_seeded** when:
✅ You have historical fraud labels  
✅ You want to leverage semi-supervised learning  
✅ Pattern features are NOT hurting performance

### Use **ensemble_ultimate** when:
✅ You're testing "everything"  
⚠️ Be careful - may be diluted by poor methods

### Use **ensemble_pattern** when:
❌ Generally avoid - pattern_features typically underperform  
⚠️ Only use if your pattern features are actually good

---

## Strengths and Limitations

### Strengths
✅ **Robustness**: Less sensitive to individual method failures  
✅ **Coverage**: Detects diverse fraud patterns  
✅ **Performance**: Often matches or beats best individual  
✅ **Interpretability**: Can explain why an account was flagged (which methods agree)

### Limitations
⚠️ **Dilution**: Poor methods can drag down performance  
⚠️ **Complexity**: Harder to interpret than single method  
⚠️ **Computation**: Must run multiple methods (slower)  
⚠️ **Weight Tuning**: Weights are somewhat arbitrary (could be optimized)

---

## Advanced Topics

### Weight Optimization

Current weights (e.g., 0.40, 0.35, 0.25) are heuristic. You could optimize them:

```python
# Potential optimization approach (not implemented)
from sklearn.linear_model import LogisticRegression

# Train weights on labeled data
X = np.column_stack([in_deg_scores, pagerank_scores, in_tx_scores])
y = fraud_labels

lr = LogisticRegression()
lr.fit(X, y)

# Use learned coefficients as weights
optimized_weights = lr.coef_[0]
```

### Stacking Ensembles

Could use ensemble scores as input to another layer:
```
Level 1: Individual methods → ensemble_top3, ensemble_diverse
Level 2: Combine ensembles → meta_ensemble
```

### Dynamic Weighting

Adjust weights based on recent performance:
```python
# If PageRank performed well in last window, increase its weight
if recent_ap['pagerank_wlog'] > threshold:
    weights['pagerank'] += 0.05
```

---

## Practical Recommendations

### For Your TCC

**Include in Report**:
1. Explanation of ensemble methodology (why they work)
2. Comparison of ensemble vs individual methods
3. Analysis showing ensemble robustness (lower variance)
4. Discussion of weight selection rationale

**Key Plots to Include**:
- Ensemble vs individual performance (from Section 4)
- Ensemble stability across windows
- Component contribution analysis

**Arguments to Make**:
- Ensembles provide more robust detection
- Less dependent on any single method's performance
- Better for operational deployment (reliability matters)

### For Production Deployment

**Recommended**: `ensemble_top3` or `ensemble_seeded`
- Simple, proven, reliable
- Easy to explain to stakeholders
- Good performance without complexity

**Backup**: Individual `pagerank_wlog` or `in_deg`
- Sometimes beat ensembles
- Simpler to compute
- Easier to debug

---

## Code Reference

### Location in tcc.py

**Main Functions**:
- `ensemble_scores()` (Lines 1429-1470): Core combination algorithm
- `create_ensemble_methods()` (Lines 1472-1520): Creates all ensemble variants

**Usage in Analysis**:
```python
# In run_analysis() around line 2377:
ensemble_methods = create_ensemble_methods(score_dict)
if ensemble_methods:
    ensemble_res = eval_scores(
        nodes, y_true_dict, ensemble_methods,
        k_fracs=K_FRACS, exclude_nodes=eval_exclude
    )
    results.update(ensemble_res)
```

---

## Summary

**What**: Combine multiple fraud detection methods using weighted averaging

**Why**: More robust, better coverage, competitive performance

**How**: Normalize scores → Apply weights → Sum → Rank

**Best**: `ensemble_top3` (general) or `ensemble_seeded` (with labels)

**Key Insight**: Multiple weaker detectors together can match or beat the strongest individual detector, with better reliability.

---

**Version**: 1.0  
**Date**: 2024  
**Related**: See `METRICS_EXPLAINED.md` for evaluation metrics
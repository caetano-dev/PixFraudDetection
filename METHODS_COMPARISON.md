# Detection Methods - Detailed Comparison

## Overview Table

| Method | AP | P@0.5% | P@1% | P@2% | P@5% | Coverage@1% | Speed | Complexity | Recommendation |
|--------|-----|--------|------|------|------|-------------|-------|------------|----------------|
| **pagerank_wlog** | **0.0272** | 7.41% | **7.32%** | 7.18% | 5.15% | 2.38% | Fast | Medium | ✅ **DEPLOY** |
| **in_deg** | **0.0260** | 5.82% | **7.32%** | 7.14% | 4.70% | **9.42%** | **Fastest** | **Lowest** | ✅ **DEPLOY** |
| **ensemble_top3** | **0.0253** | 5.82% | **7.45%** | 6.87% | 4.18% | 7.07% | Medium | Medium | ✅ **DEPLOY** |
| ensemble_diverse | 0.0255 | 7.08% | 7.08% | 5.60% | 0.076 | 7.64% | Medium | Medium | ⚠️ Consider |
| out_deg | 0.0226 | 4.83% | 4.83% | 3.75% | 4.22% | 42.16% | **Fastest** | **Lowest** | ⚠️ Backup |
| ensemble_ultimate | 0.0202 | 7.12% | 7.12% | 5.50% | 0.062 | 6.24% | Slow | High | ⚠️ Overfitted? |
| in_amt | 0.0193 | 1.58% | 1.58% | 1.17% | 0.071 | 7.07% | **Fastest** | **Lowest** | ❌ Poor |
| in_tx | 0.0187 | 6.04% | 6.04% | 4.78% | 0.071 | 8.03% | **Fastest** | **Lowest** | ⚠️ Moderate |
| out_amt | 0.0176 | 1.90% | 1.90% | 1.36% | 0.091 | 9.08% | **Fastest** | **Lowest** | ❌ Poor |
| communities_leiden | 0.0161 | 2.26% | 2.26% | 1.71% | 0.039 | 3.90% | Slow | High | ❌ Poor |
| kcore_in | 0.0161 | 5.44% | 5.44% | 4.71% | 0.056 | 5.60% | Fast | Low | ⚠️ Moderate |
| communities_louvain | 0.0159 | 2.22% | 2.22% | 1.75% | 0.039 | 3.90% | Slow | High | ❌ Poor |
| collector | 0.0150 | 0.74% | 0.74% | 0.57% | 0.013 | 1.31% | **Fastest** | **Lowest** | ❌ Poor |
| kcore_und | 0.0150 | 3.16% | 3.16% | 2.71% | 0.079 | 7.88% | Fast | Low | ❌ Poor |
| ensemble_seeded | 0.0147 | 4.76% | 4.76% | 4.38% | 0.097 | 9.76% | Medium | Medium | ⚠️ Limited |
| out_tx | 0.0144 | 2.58% | 2.58% | 2.10% | 0.248 | 24.82% | **Fastest** | **Lowest** | ❌ Poor |
| ensemble_pattern | 0.0141 | 3.90% | 3.90% | 3.20% | 0.086 | 8.58% | Slow | High | ❌ Hurts |
| kcore_out | 0.0137 | 2.18% | 2.18% | 2.02% | 0.019 | 1.85% | Fast | Low | ❌ Poor |
| distributor | 0.0132 | 0.83% | 0.83% | 0.62% | 0.038 | 3.81% | **Fastest** | **Lowest** | ❌ Poor |
| seeded_pr | 0.0126 | 5.03% | 5.03% | 5.44% | 0.366 | 36.59% | Medium | Medium | ⚠️ Needs data |
| hits_hub | 0.0125 | 1.95% | 1.95% | 1.59% | 0.168 | 16.76% | Fast | Low | ❌ No better than random |
| hits_auth | 0.0125 | 1.95% | 1.95% | 1.59% | 0.168 | 16.76% | Fast | Low | ❌ No better than random |
| **random** | **0.0125** | 1.25% | 1.25% | 1.00% | 0.010 | 1.00% | N/A | N/A | **BASELINE** |
| pattern_features | 0.0110 | 0.83% | 0.83% | 0.70% | 0.095 | 9.52% | Slow | **Highest** | ❌ **Actively Harmful** |

---

## Category Breakdown

### 🥇 Top Tier (Production Ready)
**AP > 0.025, P@1% > 7%**

#### 1. pagerank_wlog
- **Best for:** Overall detection quality
- **How it works:** PageRank weighted by log(transaction amount)
- **Pros:** Best AP, theoretically sound, catches indirect patterns
- **Cons:** Moderate compute cost
- **Use case:** Primary production detector

#### 2. in_deg  
- **Best for:** Speed + interpretability
- **How it works:** Simple count of incoming connections
- **Pros:** Fastest, simplest, tied for best P@1%, excellent coverage (9.42%)
- **Cons:** Misses indirect patterns
- **Use case:** Real-time detection, baseline comparison

#### 3. ensemble_top3
- **Best for:** Robustness
- **How it works:** Combines top 3 methods (pagerank_wlog, in_deg, out_deg)
- **Pros:** Highest P@1% (7.45%), most robust
- **Cons:** Slower than individual methods
- **Use case:** Batch processing, maximum reliability

---

### ⚠️ Second Tier (Situational Use)
**AP 0.015-0.025, P@1% 3-7%**

#### out_deg
- **AP:** 0.0226 | **P@1%:** 4.83%
- **Best for:** Distribution hub detection
- **Note:** Excellent coverage (42%) but lower precision

#### in_tx
- **AP:** 0.0187 | **P@1%:** 6.04%
- **Best for:** High-frequency receivers
- **Note:** Moderate performance, very fast

#### ensemble_diverse
- **AP:** 0.0255 | **P@1%:** 7.08%
- **Best for:** Catching diverse scheme types
- **Note:** Good but ensemble_top3 is better

#### kcore_in
- **AP:** 0.0161 | **P@1%:** 5.44%
- **Best for:** Finding tightly connected receivers
- **Note:** Interesting but outperformed by simpler in_deg

---

### ❌ Third Tier (Not Recommended)
**AP < 0.015 or no better than random**

#### Amount-based Methods (Poor)
- **in_amt:** AP=0.0193, P@1%=1.58% - Large transactions ≠ fraud
- **out_amt:** AP=0.0176, P@1%=1.90% - Even worse
- **collector:** AP=0.0150, P@1%=0.74% - Ratio doesn't help
- **distributor:** AP=0.0132, P@1%=0.83% - Worse than collector

#### Community Methods (Too Granular)
- **communities_leiden:** AP=0.0161, P@1%=2.26%
- **communities_louvain:** AP=0.0159, P@1%=2.22%
- **Problem:** 24,000+ communities for 113,000 nodes = over-fragmentation

#### HITS Algorithm (Wrong for This Problem)
- **hits_hub/auth:** AP=0.0125, P@1%=1.95%
- **Problem:** No better than random, designed for web graphs not fraud

#### Pattern Features (Actively Harmful!)
- **pattern_features:** AP=0.0110, P@1%=0.83%
- **Problem:** Hand-crafted heuristics worse than graph structure alone
- **Impact:** Degrades ensemble performance by -25.5%

---

## Special Cases

### seeded_pr (Personalized PageRank)
- **AP:** 0.0126 | **P@1%:** 5.03%
- **Status:** ⚠️ Underperforming but has potential
- **Why it's interesting:** Uses known fraud from earlier windows as seeds
- **Why it underperforms:** Only 5 windows have seed data (limited)
- **Recommendation:** Revisit when more historical labels available

### ensemble_ultimate
- **AP:** 0.0202 | **P@1%:** 7.12%
- **Status:** ⚠️ Good P@1% but lower AP suggests overfitting
- **Note:** Includes pattern features which hurt performance

### ensemble_seeded
- **AP:** 0.0147 | **P@1%:** 4.76%
- **Status:** ⚠️ Limited by poor seeded_pr performance

---

## Performance by Window Size

### 3-Day Windows
| Method | AP | P@1% |
|--------|-----|------|
| ensemble_top3 | 0.0231 | 7.20% |
| in_deg | 0.0217 | 7.49% |
| pagerank_wlog | 0.0223 | 6.74% |

### 7-Day Windows (BETTER)
| Method | AP | P@1% |
|--------|-----|------|
| pagerank_wlog | **0.0272** | **7.32%** |
| in_deg | **0.0260** | **7.32%** |
| ensemble_top3 | **0.0253** | **7.45%** |

**Finding:** Longer windows capture more context → better performance

---

## Feature Type Effectiveness

| Feature Type | Best Method | AP | Effectiveness |
|--------------|-------------|-----|---------------|
| Graph Structure | pagerank_wlog | 0.0272 | ✅ Excellent |
| Simple Degree | in_deg | 0.0260 | ✅ Excellent |
| Transaction Count | in_tx | 0.0187 | ⚠️ Moderate |
| K-Core | kcore_in | 0.0161 | ⚠️ Moderate |
| Transaction Amount | in_amt | 0.0193 | ❌ Poor |
| Heuristic Patterns | pattern_features | 0.0110 | ❌ Harmful |
| Communities | leiden/louvain | 0.0160 | ❌ Poor |
| HITS | hits_hub/auth | 0.0125 | ❌ Useless |

**Key Insight:** Graph topology > Transaction patterns > Transaction amounts

---

## Computational Complexity

| Complexity | Methods | AP Range | Deploy? |
|------------|---------|----------|---------|
| **O(N)** - Linear | in_deg, out_deg, amounts, tx_counts | 0.018-0.026 | ✅ Yes (in_deg) |
| **O(N log N)** - Fast | kcore methods | 0.014-0.016 | ⚠️ Marginal |
| **O(N·E)** - Medium | pagerank, HITS | 0.012-0.027 | ✅ Yes (pagerank) |
| **O(N²)** - Slow | communities, pattern_features | 0.011-0.016 | ❌ Not worth it |

**Winner:** in_deg (O(N) complexity, 0.026 AP) - Best speed/performance trade-off

---

## Detection Philosophy

### **Aggregation Detectors** (BEST)
Focus on accounts that collect money from many sources
- pagerank_wlog ✅
- in_deg ✅
- in_tx ⚠️

### **Distribution Detectors** (MODERATE)
Focus on accounts that send money to many destinations
- out_deg ⚠️
- out_tx ❌

### **Amount Detectors** (POOR)
Focus on transaction volumes
- in_amt ❌
- out_amt ❌

### **Pattern Detectors** (FAILED)
Look for specific fraud patterns
- pattern_features ❌
- collector/distributor ❌

### **Community Detectors** (TOO GRANULAR)
Group accounts and score groups
- louvain/leiden ❌

**Key Finding:** Money laundering is primarily an **aggregation problem**

---

## Ensemble Strategies

### Successful Ensembles
1. **ensemble_top3** (AP=0.0253)
   - Combines: pagerank_wlog + in_deg + out_deg
   - Strategy: Best of each detection type
   - Result: ✅ Works well

2. **ensemble_diverse** (AP=0.0255)
   - Strategy: Diverse detection philosophies
   - Result: ✅ Works well

### Failed Ensembles
1. **ensemble_ultimate** (AP=0.0202)
   - Problem: Includes pattern_features
   - Result: ⚠️ Dragged down

2. **ensemble_pattern** (AP=0.0141)
   - Problem: Overweights pattern_features
   - Result: ❌ Worse than individual methods

3. **ensemble_seeded** (AP=0.0147)
   - Problem: Limited seed data
   - Result: ❌ Underperforms

**Lesson:** Only ensemble methods that work individually!

---

## Deployment Decision Tree

```
START: Need to detect money laundering

├─ Need BEST performance?
│  └─> Use: pagerank_wlog (AP=0.0272)
│
├─ Need FASTEST processing?
│  └─> Use: in_deg (AP=0.0260, nearly as good)
│
├─ Need MOST ROBUST?
│  └─> Use: ensemble_top3 (AP=0.0253, P@1%=7.45%)
│
├─ Real-time constraints?
│  └─> Use: in_deg (O(N) complexity)
│
├─ Batch processing OK?
│  └─> Use: ensemble_top3 or pagerank_wlog
│
├─ Have historical labeled data?
│  └─> Revisit: seeded_pr (currently limited)
│
└─ Research project?
   └─> Try: Graph Neural Networks, better seeds
```

---

## Summary Recommendations

### ✅ TIER S - Deploy Immediately
1. **pagerank_wlog** - Best overall
2. **in_deg** - Best speed/performance
3. **ensemble_top3** - Best robustness

### ⚠️ TIER A - Consider for Specific Use Cases
4. **out_deg** - If coverage more important than precision
5. **ensemble_diverse** - Alternative ensemble

### 🔬 TIER B - Research Potential
6. **seeded_pr** - With more labeled data
7. **kcore_in** - Interesting but outperformed

### ❌ TIER F - Do Not Use
- pattern_features (actively harmful)
- Amount-only methods (not discriminative)
- HITS (wrong algorithm)
- Communities (too granular)
- Collector/distributor (worse than random)

---

**Last Updated:** Based on metrics_log_20251016_121512.txt  
**Dataset:** 123,581 accounts, 267,899 transactions, 1.25% fraud prevalence  
**Window Analysis:** 10 windows × 2 sizes (3-day and 7-day)
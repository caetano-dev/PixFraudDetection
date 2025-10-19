# Metrics Quick Reference Card

## 🎯 The 3 Metrics That Matter Most

### 1. Average Precision (AP)
**Your Score: 0.0272** | Random: 0.0125 | **Grade: B+ ✅**
- Measures overall ranking quality
- Higher = better at ranking fraud above legitimate

### 2. Precision @ 1% (P@1%)
**Your Score: 7.32%** | Random: 1.25% | **Grade: A- ✅**
- % of fraud in top 1% of flagged accounts
- Higher = fewer false positives to investigate

### 3. Attempt Coverage @ 1%
**Your Score: 9.42%** | **Grade: C+ ⚠️**
- % of fraud schemes detected by checking top 1%
- Higher = catch more criminal networks

---

## 📊 Method Leaderboard

| Rank | Method | AP | P@1% | Speed | Deploy? |
|------|--------|-----|------|-------|---------|
| 🥇 | pagerank_wlog | 0.0272 | 7.32% | Fast | ✅ YES |
| 🥈 | in_deg | 0.0260 | 7.32% | Fastest | ✅ YES |
| 🥉 | ensemble_top3 | 0.0253 | 7.45% | Medium | ✅ YES |
| 4 | ensemble_diverse | 0.0255 | 7.08% | Medium | ⚠️ Maybe |
| 5 | out_deg | 0.0226 | 4.83% | Fastest | ⚠️ Backup |
| - | **random** | **0.0125** | **1.25%** | - | **baseline** |

---

## 🚦 Performance Interpretation Guide

### Average Precision (AP)
- 🔴 < 0.015: Poor (near random)
- 🟡 0.015-0.022: Fair
- 🟢 0.022-0.030: Good ← **You are here**
- 🌟 > 0.030: Excellent

### Precision @ 1%
- 🔴 < 2%: Poor
- 🟡 2-5%: Fair
- 🟢 5-8%: Good ← **You are here**
- 🌟 > 8%: Excellent

### Lift (vs random)
- 🔴 1-2x: Marginal
- 🟡 2-4x: Moderate
- 🟢 4-8x: Good ← **You are here (6x)**
- 🌟 > 8x: Exceptional

---

## 📈 What Each Method Detects

### pagerank_wlog 🥇
**Catches:** Collection hubs that aggregate money from many sources
**Strength:** Considers indirect connections
**Weakness:** Computationally heavier

### in_deg 🥈
**Catches:** Accounts receiving from many different senders
**Strength:** Simple, fast, interpretable
**Weakness:** Ignores indirect patterns

### ensemble_top3 🥉
**Catches:** Combines multiple detection patterns
**Strength:** Most robust, fewer blind spots
**Weakness:** Slower to compute

---

## ⚡ Quick Decision Tree

**Need best performance?** → Use `pagerank_wlog`

**Need speed?** → Use `in_deg` (nearly as good)

**Need robustness?** → Use `ensemble_top3`

**Need real-time?** → Use `in_deg` only

**Have more labeled data?** → Try `seeded_pr`

---

## 🎓 Metric Definitions (Plain English)

**AP (Average Precision)**
= How good is your ranking of ALL accounts?
= Area under precision-recall curve

**P@K (Precision at K%)**
= Of the top K% you investigate, what % are actually fraud?
= True Positives / (True Positives + False Positives) in top K%

**Coverage@K**
= Of all fraud schemes, what % do you catch by checking top K%?
= Schemes with ≥1 account in top K% / Total schemes

**Lift**
= How much better than random guessing?
= Your precision / Random precision

---

## 📋 Dataset Context

- **Accounts:** 123,581
- **Transactions:** 267,899
- **Time period:** Sep 1-16, 2022
- **Fraud prevalence:** 1.25% (highly imbalanced!)
- **Windows analyzed:** 10 (3-day) + 10 (7-day)

---

## 🔍 Investigation Scenario

**You have 1,000 investigators. How to allocate?**

### Option A: Random Investigation
- Check 1% of accounts randomly (1,236 accounts)
- Find ~15 fraudsters (1.25% hit rate)
- Waste 1,221 investigations (98.75% false positive)

### Option B: Use Best Model (pagerank_wlog)
- Check top 1% by score (1,236 accounts)
- Find ~90 fraudsters (7.32% hit rate)
- Still waste 1,146 investigations (92.68% false positive)
- **6x MORE EFFICIENT** ✅

---

## ❌ Methods That DON'T Work

| Method | AP | Why It Fails |
|--------|-----|--------------|
| pattern_features | 0.0110 | Hand-crafted heuristics miss mark |
| collector ratio | 0.0150 | Legitimate businesses also collect |
| hits_hub/auth | 0.0125 | Wrong algorithm for this problem |
| in_amt only | 0.0193 | Volume ≠ fraud |

---

## 🎯 Action Items

### ✅ DO THIS NOW:
1. Deploy `pagerank_wlog` as primary detector
2. Use `in_deg` as fast backup
3. Set investigation threshold at top 1% (P@1% = 7.32%)

### 🔬 RESEARCH NEXT:
1. Why is coverage only 9%? Different scheme types?
2. Can Graph Neural Networks improve to P@1% > 10%?
3. More historical labels → better seeded PageRank?

### ❌ DON'T WASTE TIME ON:
1. Pattern feature engineering (proven to hurt)
2. Amount-only methods
3. HITS algorithm
4. Community detection (too granular)

---

## 🏆 Success Criteria Met

- ✅ AP > 2x random baseline (117% improvement)
- ✅ P@1% > 5% (achieved 7.32%)
- ✅ Consistent across time windows
- ✅ Computationally feasible
- ⚠️ Coverage < 20% (only 9%, room for improvement)

**Overall Grade: B+ to A-**

Ready for production deployment with continuous improvement pipeline.

---

**Print this page and keep it at your desk!** 📌
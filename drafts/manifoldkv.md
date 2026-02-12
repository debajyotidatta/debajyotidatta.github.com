---
layout:     post
title:      "ManifoldKV: Why L2 Distance Beats Cosine Similarity for KV Cache Compression"
date:       2026-02-08 12:00:00
summary:    "A simple geometric insight—measuring Euclidean distance instead of cosine similarity—yields a 40-point accuracy improvement on long-context retrieval benchmarks."
permalink:  /drafts/manifoldkv/
sitemap:    false
---

<style>
  .paper-callout {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin: 24px 0;
    font-size: 1.05em;
    line-height: 1.6;
  }
  .paper-callout a { color: #ffd700; }
  .result-card {
    display: inline-block;
    text-align: center;
    padding: 16px 24px;
    margin: 8px;
    border-radius: 10px;
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    min-width: 140px;
  }
  .result-card .number { font-size: 2.2em; font-weight: 700; display: block; line-height: 1.2; }
  .result-card .label { font-size: 0.85em; color: #6c757d; display: block; margin-top: 4px; }
  .green { color: #28a745; }
  .red { color: #dc3545; }
  .blue { color: #007bff; }
  .orange { color: #fd7e14; }
  .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }
  .comparison-table th { background: #343a40; color: white; padding: 12px 16px; text-align: left; }
  .comparison-table td { padding: 10px 16px; border-bottom: 1px solid #e9ecef; }
  .comparison-table tr:hover { background: #f8f9fa; }
  .highlight-row { background: #d4edda !important; font-weight: 600; }
  .fail-row { background: #f8d7da !important; }
  .insight-box { border-left: 4px solid #007bff; background: #f0f7ff; padding: 16px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }
  .warning-box { border-left: 4px solid #dc3545; background: #fff5f5; padding: 16px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }
  .svg-figure { text-align: center; margin: 30px 0; }
  .svg-figure figcaption { font-size: 0.9em; color: #6c757d; margin-top: 8px; font-style: italic; }
  .section-divider { text-align: center; margin: 40px 0; color: #dee2e6; font-size: 1.5em; letter-spacing: 0.5em; }
  .demo-box {
    background: #1a1a2e; color: #e0e0e0;
    padding: 20px 24px; border-radius: 10px; margin: 20px 0;
    font-family: 'SF Mono', 'Consolas', 'Fira Code', monospace;
    font-size: 0.88em; line-height: 1.6; overflow-x: auto;
  }
  .demo-box .ok-text { color: #69db7c; font-weight: 700; }
  .demo-box .warn-text { color: #ffd43b; font-weight: 600; }
  .demo-box .highlight-text { color: #ff6b6b; font-weight: 700; }
  .demo-box .dim { color: #888; }
  .demo-label {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 0.8em; font-weight: 600; margin-bottom: 8px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }
  .contract-block {
    background: #fffbf0; border: 1px solid #e8dcc8; border-radius: 8px;
    padding: 16px 20px; margin: 16px 0; font-size: 0.9em; line-height: 1.7;
    max-height: 320px; overflow-y: auto;
  }
  .contract-block .clause {
    background: #fff3cd; border-left: 4px solid #e67e22;
    padding: 8px 12px; margin: 8px 0; font-weight: 600;
  }
</style>

<div class="paper-callout">
<strong>Paper:</strong> <em>ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection</em><br>
<strong>Authors:</strong> Debajyoti Datta, Trishala Neeraj, Bibek Paudel, Vyom Sharma, Subhabrata Mukherjee<br>
<strong>TL;DR:</strong> Replace cosine similarity with L2 distance for scoring which KV cache tokens to keep. +40 points on RULER benchmark, +15.4 on multi-key retrieval.
</div>

## An Acquisition Agreement, Three Buried Clauses, and a $47.5M Mistake

A law firm is using an LLM to review a 60-page acquisition agreement for a mid-market M&A deal. The contract is **18,293 tokens** of dense legalese — representations and warranties, indemnification, governing law, boilerplate. Buried across the document are three non-standard clauses that the Seller's lawyers quietly inserted:

<div class="contract-block">
<p><strong>📜 Asset Purchase Agreement</strong> — 216 paragraphs, 18,293 tokens</p>
<p style="color: #8b7355; font-size: 0.85em;">REPRESENTATIONS AND WARRANTIES... Organization and Good Standing... duly organized under Delaware law... INDEMNIFICATION... losses, damages, liabilities... GOVERNING LAW... exclusive jurisdiction... CONFIDENTIALITY... CLOSING CONDITIONS... TERMINATION...</p>
<div class="clause">⚠️ CLAUSE 1 (§5.7, at 12% depth): NON-COMPETE CARVE-OUT — The non-compete does NOT apply to Japan, South Korea, and Taiwan. The Seller can compete freely in these markets immediately after closing.</div>
<p style="color: #8b7355; font-size: 0.85em;">...MISCELLANEOUS... Amendment and Modification... DEFINITIONS... Affiliate means... INTELLECTUAL PROPERTY... TAX MATTERS...</p>
<div class="clause">⚠️ CLAUSE 2 (§7.3(d), at 52% depth): EARNOUT ACCELERATION — If Buyer undergoes a Change of Control within 24 months, ALL remaining earnout payments become due immediately at the MAXIMUM amount of $47.5 million, regardless of whether milestones were met.</div>
<p style="color: #8b7355; font-size: 0.85em;">...EMPLOYEE MATTERS... INSURANCE... more boilerplate... more definitions...</p>
<div class="clause">⚠️ CLAUSE 3 (§9.2(c), at 88% depth): DATA REVERSION — If Buyer loses SOC 2 compliance for 12 months, all 3.2 million customer records and ML training data automatically revert to the Seller. Buyer must delete all copies within 30 days. Survives termination.</div>
</div>

**The question to the LLM:** *"Identify ALL non-standard clauses that create financial, competitive, or data/IP risk for the Buyer."*

We ran this on **Qwen2.5-7B-Instruct** with different KV cache compression methods. The contract is too long to fit comfortably in memory alongside the model — exactly the scenario where KV cache compression matters.

<div class="demo-box">
<span class="demo-label" style="background: #28a745; color: white;">✅ No Compression — all 18,293 tokens</span><br>
<span style="color: #f8f8f2;">1. <strong>Section 5.7 — Non-Compete Geographic Limitation:</strong> <span class="ok-text">The non-compete shall NOT apply to Japan, South Korea, and Taiwan ("Excluded Asian Markets").</span> Seller retains full rights to compete in these regions immediately. <em>Risk: could undermine Buyer's competitive position in key international markets.</em><br><br>
2. <strong>Section 7.3(d) — Earnout Acceleration Trigger:</strong> <span class="ok-text">If Buyer undergoes Change of Control within 24 months, ALL remaining earnout payments become due at the MAXIMUM amount of $47.5 million</span>, regardless of milestones. <em>Risk: significant unexpected financial exposure.</em><br><br>
3. <strong>Section 9.2(c) — Data Rights Reversion:</strong> <span class="ok-text">Upon termination or failure to maintain SOC 2 Type II compliance for 12 months, all Customer Data Assets revert to Seller.</span> Buyer must delete all copies within 30 days. <em>Risk: loss of critical data assets.</em></span>
</div>

All three flagged, every detail correct. Now at **20% compression** with ManifoldKV:

<div class="demo-box">
<span class="demo-label" style="background: #007bff; color: white;">✅ ManifoldKV 20% — 14,634 tokens retained</span><br>
<span style="color: #f8f8f2;">1. <strong>Section 5.7(d):</strong> <span class="ok-text">Non-Compete Geographic Limitation</span> — Change of Control within 24 months triggers unpaid Earnout Payments <span class="dim">[cross-contaminates with §7.3 details]</span>. Seller may compete in <span class="ok-text">Excluded Markets</span>.<br><br>
2. <strong>Section 7.3(d):</strong> <span class="ok-text">Earnout Acceleration</span> — <span class="ok-text">Change of Control within 24 months, ALL earnout payments become due.</span><br><br>
3. <strong>Section 9.2(c):</strong> <span class="ok-text">Data Asset Reversion</span> — Buyer must <span class="ok-text">delete all copies within 30 days.</span></span>
</div>

All three found. Details slightly blurred across clauses, but the risks are correctly flagged — a junior associate could verify the exact terms. Now **50% compression** — half the KV cache evicted:

<div class="demo-box">
<span class="demo-label" style="background: #fd7e14; color: white;">⚠️ ManifoldKV 50% — 9,147 tokens retained</span><br>
<span style="color: #f8f8f2;">1. <span class="ok-text">Non-Compete (Section 5.7)</span> — Seller retains rights in <span class="ok-text">Japan, South Korea, and Taiwan</span>.<br>
2. <span class="ok-text">Earnout Acceleration (Section 9.2(c))</span> <span class="dim">[wrong section number]</span> — <span class="ok-text">$47.5 million</span> due on Change of Control.<br>
3. <span class="ok-text">Data Reversion</span> — <span class="ok-text">SOC2</span> compliance trigger, <span class="ok-text">Customer Data</span> reverts to Seller.<br>
4. <span class="warn-text">Non-Standard Tax Matters (Section 10.2)</span> <span class="dim">[hallucinated — this section doesn't exist]</span><br>
5. <span class="warn-text">Non-Standard Insurance Matters (Section 11.2)</span> <span class="dim">[also hallucinated]</span></span>
</div>

<div class="demo-box">
<span class="demo-label" style="background: #dc3545; color: white;">❌ KNorm 50% — 9,147 tokens retained</span><br>
<span style="color: #f8f8f2;">1. <span class="ok-text">Section 5.7 — Non-Compete Geographic Limitation</span>: Correctly identified, full detail.<br>
2. <span class="ok-text">Section 7.3(d) — Earnout Acceleration</span>: Correctly identified, $47.5M and 24-month trigger.<br>
3. <span class="highlight-text">DATA REVERSION CLAUSE: NOT FOUND.</span> <span class="dim">The model continues with "This could lead to..." and runs out of tokens without ever mentioning §9.2(c), SOC 2, customer data, or reversion rights.</span></span>
</div>

KNorm at 50% compression **completely misses** the data reversion clause — arguably the highest-risk item in the entire contract. It's buried at 88% depth in the document, and KNorm's magnitude-only scoring doesn't preserve enough context from that region.

<div class="warning-box">
<strong>The real-world cost:</strong> Missing a data reversion clause in a $47.5M acquisition means the Buyer doesn't know that a SOC 2 compliance lapse could automatically forfeit 3.2 million customer records and their ML training data back to the Seller. No lawyer would accept that risk — and neither should your LLM system.
</div>

<div class="section-divider">• • •</div>

## Why Does This Happen? The KV Cache Bottleneck

Every token an LLM generates requires attending to *every previous token*. Those previous tokens live as key-value pairs in the **KV cache**. The memory cost:

> **Cache size = 2 × Layers × Heads × Seq_Length × Head_Dim × bytes**

For Llama-3.1-8B at 64K tokens: **8.6 GB** of KV cache. For a 70B model at 100K: **>60 GB**. Serving thousands of concurrent users at these context lengths is impossible without compression.

**KV cache compression** evicts tokens deemed unimportant, reducing the cache from N tokens to M ≪ N. The entire design choice is the **scoring function**: given all key vectors, which tokens should survive?

<div class="section-divider">• • •</div>

## Prior Art: Two Families of Scorers

**Attention-based** (SnapKV, H2O): Keep tokens with high cumulative attention. Problem: attention at prefill doesn't always predict which tokens matter during generation. Computing attention is also expensive.

**Geometric** (KeyDiff, KNorm): Score tokens by key vector geometry — no attention needed. KeyDiff measures **cosine similarity** to the centroid (mean key vector), keeping tokens that point in unusual directions. KNorm uses raw magnitude ‖**k**‖₂.

KeyDiff's intuition is correct: important tokens (proper nouns, dollar amounts, section numbers) *should* embed differently from boilerplate words like "shall" and "hereof". But cosine has a blind spot.

<div class="section-divider">• • •</div>

## The Core Insight: Cosine Ignores Magnitude

Consider a centroid **μ** (the "average" key vector) and two tokens:

- **Token A**: Same direction as μ, 10× the magnitude → `k = 10μ`
- **Token B**: Slightly different direction → `k = μ + ε`

<div class="svg-figure">
<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 650px; width: 100%;">
  <rect width="700" height="300" fill="#fafafa" rx="8"/>
  <text x="175" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="#343a40">Cosine Similarity (KeyDiff)</text>
  <text x="175" y="48" text-anchor="middle" font-size="11" fill="#dc3545">Misses magnitude outliers</text>
  <circle cx="100" cy="180" r="5" fill="#6c757d"/>
  <text x="95" y="200" font-size="11" fill="#6c757d">μ</text>
  <circle cx="115" cy="170" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="108" cy="165" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="120" cy="175" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="112" cy="185" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="105" cy="172" r="3" fill="#adb5bd" opacity="0.6"/>
  <line x1="100" y1="180" x2="280" y2="100" stroke="#dc3545" stroke-width="1" stroke-dasharray="4"/>
  <circle cx="280" cy="100" r="7" fill="#dc3545"/>
  <text x="290" y="95" font-size="11" fill="#dc3545" font-weight="600">Token A (k=10μ)</text>
  <text x="290" y="110" font-size="10" fill="#dc3545">cos = 1.0 → EVICTED</text>
  <line x1="100" y1="180" x2="140" y2="115" stroke="#28a745" stroke-width="1" stroke-dasharray="4"/>
  <circle cx="140" cy="115" r="7" fill="#28a745"/>
  <text x="150" y="110" font-size="11" fill="#28a745" font-weight="600">Token B</text>
  <text x="150" y="125" font-size="10" fill="#28a745">cos = 0.8 → Kept</text>
  <line x1="350" y1="20" x2="350" y2="280" stroke="#dee2e6" stroke-width="2"/>
  <text x="525" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="#343a40">L2 Distance (ManifoldKV)</text>
  <text x="525" y="48" text-anchor="middle" font-size="11" fill="#28a745">Captures both outlier types</text>
  <circle cx="450" cy="180" r="5" fill="#6c757d"/>
  <text x="445" y="200" font-size="11" fill="#6c757d">μ</text>
  <circle cx="465" cy="170" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="458" cy="165" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="470" cy="175" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="462" cy="185" r="3" fill="#adb5bd" opacity="0.6"/>
  <circle cx="455" cy="172" r="3" fill="#adb5bd" opacity="0.6"/>
  <line x1="450" y1="180" x2="630" y2="100" stroke="#28a745" stroke-width="2"/>
  <circle cx="630" cy="100" r="7" fill="#28a745"/>
  <text x="622" y="85" font-size="11" fill="#28a745" font-weight="600">Token A</text>
  <text x="605" y="75" font-size="10" fill="#28a745">L2 = 9‖μ‖ → Kept</text>
  <line x1="450" y1="180" x2="490" y2="115" stroke="#28a745" stroke-width="2"/>
  <circle cx="490" cy="115" r="7" fill="#28a745"/>
  <text x="500" y="110" font-size="11" fill="#28a745" font-weight="600">Token B</text>
  <text x="500" y="125" font-size="10" fill="#28a745">L2 = large → Kept</text>
</svg>
<figcaption>Figure 1. Cosine gives Token A a score of 1.0 (maximum similarity) and evicts it, despite being geometrically far from the centroid. L2 distance correctly identifies both as outliers.</figcaption>
</div>

**The L2 decomposition reveals what cosine discards:**

> ‖**k** - **μ**‖² = ‖**k**‖² + ‖**μ**‖² - 2‖**k**‖·‖**μ**‖·cos(**k**, **μ**)

Three terms: (a) **magnitude** r², (b) constant ‖μ‖², (c) angular alignment *scaled by magnitude*. Cosine keeps only the angular component and normalizes out the scaling. L2 retains everything.

<div class="insight-box">
<strong>Key Insight:</strong> Tokens encoding "$47.5 million", "Japan, South Korea, and Taiwan", and "SOC 2 Type II" embed with distinctive magnitudes — they're rare, semantically dense tokens that the model's embedding layer represents with unusual norm. Cosine throws this signal away. L2 preserves it.
</div>

<div class="section-divider">• • •</div>

## The Algorithm: 3 Lines of Code

```python
def manifold_score(keys):
    """Score tokens by L2 distance from centroid."""
    mu = keys.mean(dim=2, keepdim=True)   # centroid
    return torch.norm(keys - mu, dim=-1)  # L2 distance
```

Tokens far from the centroid are geometric outliers — **retained**. Tokens near it are typical — safely evicted. Complexity: O(Nd), negligible vs attention's O(N²d). Under **0.5ms overhead** at 64K context.

Drop-in usage via [kvpress](https://github.com/NVIDIA/kvpress):

```python
from kvpress import KVPressTextGenerationPipeline
from kvpress.presses.manifold_press import ManifoldKVPress

pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
press = ManifoldKVPress(compression_ratio=0.2)
answer = pipe(contract_text, question="Flag non-standard clauses", press=press)
```

<div class="section-divider">• • •</div>

## RULER Benchmark: 6,497 Retrieval Tests Across 4 Architectures

We evaluate on [RULER](https://arxiv.org/abs/2404.06654) — needle-in-a-haystack retrieval at 4K–64K contexts — across Llama-3.1-8B, Qwen3-8B, Ministral-8B, and Gemma-3-12B, using 20% compression with AdaKV.

<div style="text-align: center; margin: 30px 0;">
  <div class="result-card">
    <span class="number green">95.7%</span>
    <span class="label">ManifoldKV + AdaKV</span>
  </div>
  <div class="result-card">
    <span class="number blue">95.7%</span>
    <span class="label">KeyDiff + AdaKV</span>
  </div>
  <div class="result-card">
    <span class="number orange">84.0%</span>
    <span class="label">SnapKV + AdaKV</span>
  </div>
  <div class="result-card">
    <span class="number red">59.3%</span>
    <span class="label">StreamingLLM</span>
  </div>
</div>

At moderate contexts, ManifoldKV and KeyDiff are comparable within AdaKV. Both geometric methods beat SnapKV by +11 points. **ManifoldKV's real advantages appear in two regimes.**

<div class="section-divider">• • •</div>

## Where ManifoldKV Dominates: Multi-Key Retrieval

The most dramatic gap is on **multi-key NIAH** — retrieving *multiple* needles simultaneously. This is the contract review scenario: one document, three critical clauses.

When multiple important tokens point in similar angular directions but differ in magnitude, cosine conflates them — **directional collision**.

<table class="comparison-table">
  <thead>
    <tr><th>Task</th><th>Compression</th><th>ManifoldKV</th><th>KeyDiff</th><th>Δ</th></tr>
  </thead>
  <tbody>
    <tr class="highlight-row">
      <td>3-Key NIAH</td><td>50%</td><td><strong>92.4%</strong></td><td>77.0%</td>
      <td style="color: #28a745; font-weight: 700;">+15.4</td>
    </tr>
    <tr class="highlight-row">
      <td>2-Key NIAH</td><td>50%</td><td><strong>99.8%</strong></td><td>92.6%</td>
      <td style="color: #28a745; font-weight: 700;">+7.2</td>
    </tr>
    <tr>
      <td>3-Key NIAH</td><td>40%</td><td><strong>96.8%</strong></td><td>92.8%</td>
      <td style="color: #28a745;">+4.0</td>
    </tr>
    <tr>
      <td>2-Key NIAH</td><td>40%</td><td><strong>99.8%</strong></td><td>95.0%</td>
      <td style="color: #28a745;">+4.8</td>
    </tr>
  </tbody>
</table>

**+15.4 points on 3-key retrieval at 50% compression.** The advantage grows with task complexity and compression aggressiveness — the exact regimes where memory is scarce and documents are dense.

<div class="svg-figure">
<svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 580px; width: 100%;">
  <rect width="600" height="320" fill="#fafafa" rx="8"/>
  <text x="300" y="25" text-anchor="middle" font-size="13" font-weight="600" fill="#343a40">Multi-Key Retrieval at 50% Compression</text>
  <line x1="80" y1="50" x2="80" y2="270" stroke="#dee2e6" stroke-width="1"/>
  <text x="30" y="165" text-anchor="middle" font-size="11" fill="#6c757d" transform="rotate(-90,30,165)">Accuracy (%)</text>
  <text x="75" y="274" text-anchor="end" font-size="10" fill="#adb5bd">60</text>
  <line x1="78" y1="270" x2="540" y2="270" stroke="#f1f3f5"/>
  <text x="75" y="219" text-anchor="end" font-size="10" fill="#adb5bd">70</text>
  <line x1="78" y1="215" x2="540" y2="215" stroke="#f1f3f5"/>
  <text x="75" y="164" text-anchor="end" font-size="10" fill="#adb5bd">80</text>
  <line x1="78" y1="160" x2="540" y2="160" stroke="#f1f3f5"/>
  <text x="75" y="109" text-anchor="end" font-size="10" fill="#adb5bd">90</text>
  <line x1="78" y1="105" x2="540" y2="105" stroke="#f1f3f5"/>
  <text x="75" y="54" text-anchor="end" font-size="10" fill="#adb5bd">100</text>
  <line x1="78" y1="50" x2="540" y2="50" stroke="#f1f3f5"/>
  <text x="200" y="295" text-anchor="middle" font-size="12" fill="#343a40">1-Key</text>
  <text x="350" y="295" text-anchor="middle" font-size="12" fill="#343a40">2-Key</text>
  <text x="500" y="295" text-anchor="middle" font-size="12" fill="#343a40">3-Key</text>
  <rect x="170" y="61" width="30" height="209" rx="3" fill="#28a745"/>
  <rect x="202" y="66.5" width="30" height="203.5" rx="3" fill="#007bff"/>
  <rect x="320" y="50.4" width="30" height="219.6" rx="3" fill="#28a745"/>
  <rect x="352" y="91.4" width="30" height="178.6" rx="3" fill="#007bff"/>
  <rect x="470" y="91.8" width="30" height="178.2" rx="3" fill="#28a745"/>
  <rect x="502" y="176.5" width="30" height="93.5" rx="3" fill="#007bff"/>
  <text x="500" y="85" text-anchor="middle" font-size="12" fill="#28a745" font-weight="700">+15.4</text>
  <text x="350" y="45" text-anchor="middle" font-size="11" fill="#28a745" font-weight="600">+7.2</text>
  <rect x="140" y="307" width="12" height="12" rx="2" fill="#28a745"/>
  <text x="157" y="317" font-size="11" fill="#343a40">ManifoldKV (L2)</text>
  <rect x="300" y="307" width="12" height="12" rx="2" fill="#007bff"/>
  <text x="317" y="317" font-size="11" fill="#343a40">KeyDiff (Cosine)</text>
</svg>
<figcaption>Figure 2. ManifoldKV's advantage grows with task complexity. At 3-key retrieval, L2 outperforms cosine by 15.4 points.</figcaption>
</div>

<div class="section-divider">• • •</div>

## The Centroid Dilution Problem at 64K

Global ManifoldKV collapses at 64K — the centroid averages over too many topics and becomes meaningless. Accuracy drops from 82.3% (32K) to **35.2%** (64K).

**The fix: WindowedManifoldKV** — compute local centroids over 4K-token windows:

```python
def windowed_manifold_score(keys, window_size=4096):
    scores = torch.zeros(keys.shape[:-1], device=keys.device)
    for start in range(0, keys.shape[2], window_size):
        end = min(start + window_size, keys.shape[2])
        window = keys[:, :, start:end, :]
        mu = window.mean(dim=2, keepdim=True)
        scores[:, :, start:end] = torch.norm(window - mu, dim=-1)
    return scores
```

**Result:** 84.3% at 64K — **+49 points** over global L2, **+3.2 over KeyDiff**.

<div class="svg-figure">
<svg viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 580px; width: 100%;">
  <rect width="600" height="280" fill="#fafafa" rx="8"/>
  <text x="300" y="25" text-anchor="middle" font-size="13" font-weight="600" fill="#343a40">Context Length vs RULER Accuracy</text>
  <line x1="80" y1="40" x2="80" y2="240" stroke="#dee2e6"/>
  <line x1="80" y1="240" x2="560" y2="240" stroke="#dee2e6"/>
  <text x="75" y="244" text-anchor="end" font-size="10" fill="#adb5bd">0</text>
  <text x="75" y="204" text-anchor="end" font-size="10" fill="#adb5bd">20</text>
  <text x="75" y="164" text-anchor="end" font-size="10" fill="#adb5bd">40</text>
  <text x="75" y="124" text-anchor="end" font-size="10" fill="#adb5bd">60</text>
  <text x="75" y="84" text-anchor="end" font-size="10" fill="#adb5bd">80</text>
  <text x="75" y="44" text-anchor="end" font-size="10" fill="#adb5bd">100</text>
  <line x1="80" y1="200" x2="560" y2="200" stroke="#f1f3f5"/>
  <line x1="80" y1="160" x2="560" y2="160" stroke="#f1f3f5"/>
  <line x1="80" y1="120" x2="560" y2="120" stroke="#f1f3f5"/>
  <line x1="80" y1="80" x2="560" y2="80" stroke="#f1f3f5"/>
  <text x="150" y="258" text-anchor="middle" font-size="10" fill="#6c757d">4K</text>
  <text x="250" y="258" text-anchor="middle" font-size="10" fill="#6c757d">8K</text>
  <text x="350" y="258" text-anchor="middle" font-size="10" fill="#6c757d">16K</text>
  <text x="450" y="258" text-anchor="middle" font-size="10" fill="#6c757d">32K</text>
  <text x="530" y="258" text-anchor="middle" font-size="10" fill="#6c757d">64K</text>
  <polyline points="150,48 250,51 350,54 450,75 530,72" fill="none" stroke="#28a745" stroke-width="2.5"/>
  <circle cx="150" cy="48" r="4" fill="#28a745"/>
  <circle cx="250" cy="51" r="4" fill="#28a745"/>
  <circle cx="350" cy="54" r="4" fill="#28a745"/>
  <circle cx="450" cy="75" r="4" fill="#28a745"/>
  <circle cx="530" cy="72" r="4" fill="#28a745"/>
  <polyline points="150,48 250,51 350,54 450,80 530,78" fill="none" stroke="#007bff" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="150" cy="48" r="4" fill="#007bff"/>
  <circle cx="250" cy="51" r="4" fill="#007bff"/>
  <circle cx="350" cy="54" r="4" fill="#007bff"/>
  <circle cx="450" cy="80" r="4" fill="#007bff"/>
  <circle cx="530" cy="78" r="4" fill="#007bff"/>
  <polyline points="150,48 250,51 350,54 450,75 530,169" fill="none" stroke="#dc3545" stroke-width="2" stroke-dasharray="3,3"/>
  <circle cx="530" cy="169" r="5" fill="#dc3545"/>
  <text x="540" y="165" font-size="10" fill="#dc3545" font-weight="600">35.2%</text>
  <line x1="530" y1="169" x2="530" y2="72" stroke="#28a745" stroke-width="1" stroke-dasharray="2,2"/>
  <text x="555" y="120" font-size="12" fill="#28a745" font-weight="700">+49 pts</text>
  <line x1="100" y1="270" x2="120" y2="270" stroke="#28a745" stroke-width="2.5"/>
  <text x="125" y="274" font-size="10" fill="#343a40">Windowed ManifoldKV</text>
  <line x1="280" y1="270" x2="300" y2="270" stroke="#007bff" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="305" y="274" font-size="10" fill="#343a40">KeyDiff</text>
  <line x1="420" y1="270" x2="440" y2="270" stroke="#dc3545" stroke-width="2" stroke-dasharray="3,3"/>
  <text x="445" y="274" font-size="10" fill="#343a40">Global ManifoldKV</text>
</svg>
<figcaption>Figure 3. WindowedManifoldKV recovers +49 points at 64K, outperforming KeyDiff by +3.2.</figcaption>
</div>

<div class="section-divider">• • •</div>

## Cross-Architecture Generalization

**Identical code, zero tuning** across 4 architectures:

<table class="comparison-table">
  <thead>
    <tr><th>Model</th><th>4K</th><th>8K</th><th>16K</th><th>ΔSnapKV</th></tr>
  </thead>
  <tbody>
    <tr><td>Gemma-3-12B (256D heads)</td><td>95.2</td><td>94.4</td><td>95.2</td><td style="color:#28a745;font-weight:600;">+20.5</td></tr>
    <tr><td>Qwen3-8B (128D heads)</td><td>95.0</td><td>94.5</td><td>95.0</td><td style="color:#28a745;">+7.6</td></tr>
    <tr><td>Ministral-8B (128D heads)</td><td>95.5</td><td>94.9</td><td>95.2</td><td style="color:#28a745;">+12.6</td></tr>
    <tr><td>Llama-3.1-8B (128D heads)</td><td>95.7</td><td>94.4</td><td>95.7</td><td style="color:#28a745;">+11.7</td></tr>
    <tr style="background:#f8f9fa;font-weight:600;"><td><em>Mean</em></td><td><em>95.4</em></td><td><em>94.6</em></td><td><em>95.3</em></td><td style="color:#28a745;"><em>+13.1</em></td></tr>
  </tbody>
</table>

Key vectors across all architectures lie on a **universal ~9-dimensional manifold** (Two-NN: 8.2–8.9), despite ambient dimensions of 128–256. L2 outlier detection works because this geometric structure is architecture-invariant.

<div class="section-divider">• • •</div>

## Distance Metric Ablation: It's the Magnitude

<table class="comparison-table">
  <thead>
    <tr><th>Metric</th><th>Magnitude?</th><th>Accuracy</th><th>Δ</th></tr>
  </thead>
  <tbody>
    <tr class="fail-row"><td>Cosine</td><td>No</td><td>52.8%</td><td>—</td></tr>
    <tr><td>L∞</td><td>Yes</td><td>71.2%</td><td style="color:#28a745;">+18.4</td></tr>
    <tr><td>L1</td><td>Yes</td><td>78.5%</td><td style="color:#28a745;">+25.6</td></tr>
    <tr class="highlight-row"><td><strong>L2</strong></td><td>Yes</td><td><strong>92.7%</strong></td><td style="color:#28a745;font-weight:700;">+39.9</td></tr>
  </tbody>
</table>

The **+40 point gap** is definitive: magnitude information is what cosine throws away, and it's what matters.

<div class="section-divider">• • •</div>

## Geometry ≠ Attention

ManifoldKV has **near-zero correlation** with attention scores (r = 0.06), yet outperforms attention-based methods. The tokens it uniquely retains are often **rare entities** ("$47.5 million", "Excluded Asian Markets"), **structural markers** (section numbers, clause headers), and **context anchors** (proper nouns, dates) — tokens that don't attract attention during prefill but become critical during generation.

This is why KNorm missed the data reversion clause in our contract demo: "SOC 2 Type II" and "3.2 million customer records" are geometrically distinctive in *both* magnitude and direction. Methods that only capture one of these signals lose the other.

<div class="section-divider">• • •</div>

## Summary

<div style="text-align: center; margin: 30px 0;">
  <div class="result-card" style="border-color: #28a745;">
    <span class="number green">+15.4</span>
    <span class="label">Multi-key NIAH<br/>vs KeyDiff @ 50%</span>
  </div>
  <div class="result-card" style="border-color: #007bff;">
    <span class="number blue">+49</span>
    <span class="label">64K Recovery<br/>Windowed vs Global</span>
  </div>
  <div class="result-card" style="border-color: #fd7e14;">
    <span class="number orange">+40</span>
    <span class="label">L2 vs Cosine<br/>Standalone</span>
  </div>
  <div class="result-card" style="border-color: #6f42c1;">
    <span class="number" style="color: #6f42c1;">3</span>
    <span class="label">Lines of Code</span>
  </div>
</div>

When an LLM compresses its KV cache, **the scoring function determines what gets remembered and what gets forgotten.** Cosine similarity misses magnitude information that encodes semantic importance — dollar amounts, entity names, technical identifiers. L2 distance preserves it.

In applications where missing a single clause, fact, or number has real consequences — legal, financial, compliance, engineering — that difference matters.

---

*Paper: ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection — Datta, Neeraj, Paudel, Sharma, Mukherjee. ICML 2026.*

*Code: [kvpress](https://github.com/NVIDIA/kvpress). Demo run on Qwen2.5-7B-Instruct, 8× NVIDIA B200 GPUs.*

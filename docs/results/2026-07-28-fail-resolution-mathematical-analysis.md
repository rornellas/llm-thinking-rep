# Mathematical diagnosis of the pre-Qwen FAIL and next hypotheses

**Status:** post-v1.3 exploratory synthesis.  
**Frozen decision:** `NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.

## 1. What failed mathematically

For one projection, stack the flattened expert weights as

\[
\mathcal W =
\begin{bmatrix}
\mathrm{vec}(W_1)^\top\\
\vdots\\
\mathrm{vec}(W_E)^\top
\end{bmatrix}
\in \mathbb R^{E\times mn}.
\]

The scalar Modal form

\[
W_e = W_0 + \sum_{k=1}^{K} a_{e,k} W_k
\]

imposes

\[
\mathrm{rank}(\mathcal W) \le K+1.
\]

By the Eckart–Young theorem, the best unweighted rank-\((K+1)\) approximation has unavoidable squared error

\[
\min_{\mathrm{rank}(\widehat{\mathcal W})\le K+1}
\|\mathcal W-\widehat{\mathcal W}\|_F^2
=
\sum_{j>K+1}\sigma_j^2.
\]

Training and closed-loop distillation can move the student to a better function-space solution, but cannot remove the dimensional restriction. The longer-budget rank curve shows the practical consequence: the error declines as K rises, yet no K satisfies the fidelity UCB gate; K7–K8 are closest.

For top-\(T\) routing, the dominant shared-mode matrix ratio is approximately

\[
C_{\mathrm{scalar}}(K) \approx \frac{K+1}{T}.
\]

Strict matrix-compute reduction requires \(K+1<T\), or \(K\le T-2\). With \(T=4\), only K0–K2 reduce the dominant matrix count. Those are precisely the ranks that fail strongly in the longer-budget screen. K7–K8 use roughly 2.0–2.25 times the original dominant matrix arithmetic before coefficient overhead.

**Inference:** the blocking problem is not only optimization. A single global basis on the expert axis is the wrong inductive bias for the tested conventional teachers.

## 2. Why the tested repairs did not solve it

### 2.1 Expert-specific low-rank residuals

The tested family was

\[
W_e = W_e^{\mathrm{Modal}} + A_eB_e,
\qquad \mathrm{rank}(A_eB_e)\le r.
\]

This restores expert-specific directions and improves the parameter/quality frontier. But every routed expert now requires residual projections. For the tested shapes, the analytic adjusted cost was

\[
\frac{K+1}{T} + \frac{K}{d} + \frac{r(d+f)}{df}.
\]

The primary K1/R3 candidate reached 36.77% of full expert parameters and 74.17% adjusted arithmetic, but remained worse than the 75%-width conventional baseline at similar dominant matrix compute. The residual is useful capacity, but its current factorization is not more arithmetic-efficient than ordinary width.

### 2.2 Router-semantic clustered bases

The clustered family replaces one global basis by G group bases plus per-expert residuals. Its dominant shared matrix term scales as

\[
C_{\mathrm{groups}} \approx \frac{G}{T}.
\]

Clustering can reduce conflict only when routed experts within each group truly share a functionally aligned subspace. In the present screen, G2/G3 candidates did not beat the parameter-similar narrow baseline and remained far behind the matrix-matched baseline. Static router-logit clustering therefore did not create enough alignment to compensate for the extra bases.

## 3. The next function class should be alignment-tolerant

The evidence points away from sharing complete raw matrices with only scalar expert coefficients. A better class should preserve expert-specific input/output coordinates while sharing only the large subspaces that are actually common.

### Hypothesis A — expert-specific side factor plus shared basis bank

Use a MoBE-like form

\[
W_e = A_e B_e,
\qquad
B_e = \sum_{k=1}^{K} \alpha_{e,k} B_k,
\]

where the smaller side factor \(A_e\) remains expert-specific and the larger factor is a combination of shared bases. This is less sensitive to expert neuron permutations than sharing the whole matrix. MoBE reports this design for MoE compression (arXiv:2508.05257). Sub-MoE similarly shares one subspace factor while retaining expert-specific components and clusters experts by output similarity (arXiv:2506.23266).

**Test A1:** compare shared-left, shared-right, and two-sided Tucker/core variants under exactly matched parameters and analytic MACs. Primary endpoint is teacher KL; token CE is secondary.

### Hypothesis B — activation/Fisher-weighted factorization

Raw Frobenius error treats every input direction equally. The functional objective for an input covariance \(\Sigma_x\) is

\[
\mathbb E\| (W_e-\widehat W_e)x\|_2^2
=
\mathrm{tr}\left[(W_e-\widehat W_e)\Sigma_x(W_e-\widehat W_e)^\top\right].
\]

Therefore, modes should be learned after whitening or weighting by real routed activations. ASVD uses activation-aware transformations for low-rank LLM compression (arXiv:2312.05821). D²-MoE uses Fisher information to form a shared base and compresses expert deltas (arXiv:2502.17298).

**Test B1:** build per-expert routed activation covariance estimates on train-only calibration data; initialize and train the factorization in the covariance-weighted metric; evaluate on document-disjoint natural/OOD data.

### Hypothesis C — train the compact constraint before specialization hardens

The post-hoc task asks a restricted family to reproduce a function learned without that restriction. μMoE demonstrates implicit factorized expert computation when the factorization is part of training (arXiv:2402.12550). MoSE trains nested/slimmable experts under sparse routing rather than compressing a fixed mature expert after the fact (arXiv:2602.06154).

**Test C1:** train a conventional and a factorized MoE from identical initialization budgets on the same natural corpus, then perform whole-model distillation from the conventional teacher. This answers whether the compact family can learn an equally useful solution, separately from whether it can exactly transplant arbitrary mature experts.

## 4. Experimental order

1. Train the controlled conventional teachers to an explicit plateau criterion; do not call them mature before it is met.
2. Replace generated templates with document-disjoint natural text plus code, mathematics and Portuguese.
3. Run A1 and B1 with three development seeds and identical batches.
4. Compare against both parameter-matched and MAC-matched conventional widths.
5. Use KL/logit fidelity as the primary endpoint; report CE separately to detect task adaptation that departs from the teacher.
6. Only a candidate that passes absolute teacher fidelity and both matched baselines enters a fresh sealed five-seed replication.
7. Only then test one real OLMoE layer; Qwen remains downstream of that gate.

## 5. Proposed hard gates for the next development protocol

```text
teacher plateau:
  final-20%-of-training validation improvement < 0.01 nat
  and no monotone downward trend across the last three checkpoints

absolute fidelity:
  crossed-bootstrap UCB95 of ΔCE <= +0.010 nat
  crossed-bootstrap UCB95 of teacher->student KL <= +0.015 nat

matched controls:
  UCB95(student - parameter-matched) <= 0
  UCB95(student - MAC-matched) <= +0.003 nat

robustness:
  all seeds within +0.020 nat of teacher
  no domain mean above +0.025 nat
  natural OOD KL gate passes

cost:
  total expert parameters, routed MACs, code/gather cost and measured latency reported separately
```

These thresholds are development gates. Confirmation requires a new preregistration, more seeds, and a holdout that has not been used anywhere in the current branch.

## 6. Literature anchors

- Oldfield et al., *Multilinear Mixture of Experts*, arXiv:2402.12550.
- Yuan et al., *ASVD: Activation-aware Singular Value Decomposition*, arXiv:2312.05821.
- Gu et al., *Delta Decompression for MoE-based LLMs Compression (D²-MoE)*, arXiv:2502.17298.
- Li et al., *Sub-MoE*, arXiv:2506.23266.
- Chen et al., *MoBE*, arXiv:2508.05257.
- Tastan et al., *MoSE*, arXiv:2602.06154.

These papers support the design space; they do not validate our candidate or establish novelty.

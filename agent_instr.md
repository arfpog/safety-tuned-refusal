# Helpful to Whom? Differential Treatment in Safety-Aligned Language Models

An investigation into how safety-aligned LLMs treat different identity groups differently—not through refusal, but through unsolicited "helpful" additions like crisis resources and disclaimers.

## Overview

This project examines whether safety-aligned language models encode differential treatment across identity groups, and whether standard interpretability tools can detect it.

**Key Finding:** LLaMA-3-8B-Instruct systematically adds more safety language (crisis hotlines, support resources, disclaimers) to responses for certain identity groups—even for identical questions. A gay teenager asking about dating gets Trevor Project links; a straight teenager gets direct advice. However, linear probes measuring geometric entanglement between identity and safety directions find near-zero cosine similarity. The bias is *contextual*, not *geometric*—and our standard interpretability tools miss it entirely.

## Research Questions

1. **Behavioral:** Do safety-aligned LLMs treat different identity groups differently?
2. **Representational:** Where in the network is identity encoded? Where is safety behavior encoded?
3. **Geometric:** Are identity and safety directions entangled in representation space?
4. **Methodological:** Can standard linear probes detect differential treatment?

## Key Results

### Behavioral Findings

| Axis | Hedging Rate (safety prompts) |
|------|-------------------------------|
| Sexual Orientation | 96% |
| Gender Identity | 71% |
| Race | 67% |
| Religion | 54% |

Within-axis disparities:
- Trans women (88%) vs cis women (62%)
- Atheists (75%) vs Christians (38%)
- Latinos (88%) vs white students (50%)

### Interpretability Findings

| Probe | Result |
|-------|--------|
| Identity decodable? | Yes—95-100% accuracy by layer 15 |
| Safety behavior decodable? | Yes—85% accuracy from prompt representations |
| Identity-safety cosine overlap | **~0 across all layers** |

**The disconnect:** Behavioral analysis shows clear differential treatment. Geometric analysis shows no entanglement. Both are correct—the bias is contextual, not geometric.


## Future Work

### High Priority

#### 1. Expand Dataset
**Current:** 240 prompts (~20 per identity)
**Target:** 500+ prompts (~40+ per identity)

Tasks:
- [ ] Add 4 more paraphrases per scenario (use LLM to generate)
- [ ] Add 1 more scenario per axis
- [ ] Re-run full pipeline on expanded dataset

#### 2. Validate LLM Judge
**Current:** Gemini labels with no human validation
**Target:** Human-validated ground truth

Tasks:
- [ ] Manually label 50 randomly sampled responses
- [ ] Compute Cohen's Kappa agreement with Gemini
- [ ] Document systematic disagreements
- [ ] Consider refining judge prompt if agreement is low

#### 3. Add Confidence Intervals
**Current:** Point estimates only
**Target:** Bootstrap 95% CIs on all behavioral metrics

Tasks:
- [ ] Implement bootstrap CI function
- [ ] Add CIs to safety rate tables
- [ ] Add error bars to plots
- [ ] Explicitly state which findings are/aren't statistically robust

### Medium Priority

#### 4. Test Additional Models
**Goal:** Determine if findings generalize beyond LLaMA-3

Tasks:
- [ ] Run same prompts through Mistral-7B-Instruct
- [ ] Run same prompts through Gemma-2-9B-Instruct
- [ ] Compare behavioral patterns across models
- [ ] If consistent: systemic issue; if different: model-specific RLHF

#### 5. Base Model Comparison
**Goal:** Identify source of differential treatment (pretraining vs RLHF)

Tasks:
- [ ] Run prompts through base LLaMA-3-8B (not Instruct)
- [ ] Compare hedging keyword rates
- [ ] If base model shows no differential treatment → bias from safety training
- [ ] If base model shows same patterns → bias from pretraining data

#### 6. Axis-Level Cosine Analysis
**Goal:** Test whether axis membership (not just identity) correlates with safety

Tasks:
- [ ] Train probe to classify axis (SO vs GI vs REL vs RACE)
- [ ] Compute cosine between "is SO axis" direction and safety direction
- [ ] This tests whether the model associates entire topics (not identities) with safety

### Lower Priority

#### 7. Non-Lexical Identity Probes
**Goal:** Test if probes detect abstract identity or just tokens

Tasks:
- [ ] Create 20 prompts with implied identity (no explicit terms)
  - e.g., "I want to tell my parents about my boyfriend" (implies gay male)
- [ ] Test if identity probes still work
- [ ] If they fail → probes detect tokens, not concepts

#### 8. User Study
**Goal:** Understand if affected users experience this as helpful or stigmatizing

Tasks:
- [ ] Design survey showing response pairs
- [ ] Recruit LGBTQ+ participants
- [ ] Ask: "Which response would you prefer? Why?"
- [ ] This moves from technical finding to actual harm assessment

#### 9. Causal Intervention
**Goal:** Test if safety direction causally influences behavior

Tasks:
- [ ] Implement activation patching or steering
- [ ] Add/subtract safety direction from representations
- [ ] Measure if hedging behavior changes
- [ ] This would strengthen "the model uses this direction" claims

## Known Limitations

1. **Small dataset:** ~20 samples per identity means high variance
2. **Single model:** Results may be LLaMA-specific
3. **Lexical identity:** Probes likely detect tokens, not abstract concepts
4. **Unvalidated judge:** Gemini's labels may not match human judgment
5. **No user study:** Unknown if differential treatment is experienced as harmful
6. **Linear probes only:** Nonlinear patterns would be missed

## Citation

```bibtex
@misc{pogosian2025helpful,
  title={Helpful to Whom? Differential Treatment in Safety-Aligned Language Models},
  author={Pogosian, Arthur},
  year={2025},
  note={CIS 7000: Algorithmic Justice, University of Pennsylvania}
}
```

## References

- Abid, A., Farooqi, M., & Zou, J. (2021). Persistent Anti-Muslim Bias in Large Language Models.
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots.
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances.
- Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.
- Costanza-Chock, S. (2020). Design Justice.
- Dias Oliva, T., Antonialli, D. M., & Gomes, A. (2021). Fighting Hate Speech, Silencing Drag Queens?
- Hoffmann, A. L. (2019). Where Fairness Fails: Data, Algorithms, and the Limits of Antidiscrimination Discourse.
- Sap, M., et al. (2019). The Risk of Racial Bias in Hate Speech Detection.

## License

MIT License
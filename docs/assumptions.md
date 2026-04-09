# Assumptions & Validity Limits

## What This Project Is

A **simulation-based human factors evaluation** of three UI variants for satellite operations event triage. Uses synthetic data, computational operator models, and Monte Carlo analysis to estimate how interface design choices affect errors, workload, and trust under varying conditions.

## What This Project Is NOT

- **Not** a usability study with real operators
- **Not** validated against real satellite telemetry
- **Not** a production-ready decision support system
- **Not** a claim about specific satellite subsystems

## Key Assumptions

### Synthetic Data
- Event features are generated from a known latent-risk model. Real telemetry would have different distributions, correlations, and temporal dependencies.
- Log text is template-based. Real logs would be more varied and context-dependent.
- Ground truth (severity, correct action) is deterministic. Real operations involve ambiguity where experts may disagree.

### Operator Models
- Cognitive parameters (error rate, attention capacity, automation reliance) are estimated from human factors literature, not measured from real operators.
- The decision rule is a simplified mechanistic model. Real human decision-making involves additional factors: domain knowledge, team communication, emotional state, organizational culture.
- Fatigue growth is linear. Real fatigue follows circadian patterns and is modulated by task engagement.

### UI Effects
- Each UI feature is assumed to have a specific, independent effect on operator behavior. In practice, interactions between features may be complex and non-linear.
- The effect magnitudes (e.g., +0.25 verification propensity for evidence panel) are estimated, not measured.
- We assume operators actually use the features when present. Real adoption depends on training, trust, and interface discoverability.

### ML Pipeline
- The classifier is trained on synthetic data from the routine distribution. Performance on incident scenarios represents a form of distribution shift.
- Feature space is deliberately simple (5 features). Real satellite telemetry would involve many more signals and temporal patterns.

## What Claims ARE Supported

1. **Relative comparisons**: "Variant C reduces catastrophic errors compared to A" is a valid claim within the simulation framework, given the stated assumptions and parameters.
2. **Mechanism identification**: "Evidence panels reduce inappropriate reliance on weak-evidence recommendations" follows from the mechanistic model.
3. **Sensitivity ranges**: "The benefit of Variant C is robust to ±20% variation in automation reliance" demonstrates parameter robustness.
4. **Hypothesis generation**: Results guide where real-user testing should focus.

## What Claims Are NOT Supported

1. Absolute error rates or decision times for real operators
2. Comparative UI performance in real satellite operations
3. Generalization to non-triage tasks or non-space domains
4. Specific ROI or cost savings estimates

## Recommended Next Steps for Validation

1. **Wizard-of-Oz study** with 6–10 domain-adjacent participants using the web prototype
2. **Expert review** of operator model parameters by satellite ops SMEs
3. **Ecological validity check** against real telemetry patterns from mission logs
4. **Longitudinal evaluation** of fatigue and learning effects over multi-shift periods

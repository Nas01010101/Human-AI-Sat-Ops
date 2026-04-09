# Requirements Document

## User Profiles

### 1. Novice Operator
- New to console operations, slower processing
- Higher cognitive load under event pressure
- More likely to follow AI recommendations without verification
- Limited pattern recognition for anomaly signals

### 2. Experienced Operator
- Strong pattern recognition and faster decision-making
- Moderate automation trust calibrated by experience
- Higher verification propensity when evidence is available
- May over-trust automation in "it usually works" mode

### 3. On-Call / Fatigued Operator
- Interruption-prone due to concurrent responsibilities
- Slip rate increases with shift duration (fatigue growth)
- Alert fatigue from sustained high-volume monitoring
- Checklist adherence decays under time pressure

## Usage Scenarios

### Routine Shift
- 30 events over standard monitoring period
- 10% high-severity events
- Low environmental noise
- Low interruption rate

### Incident Spike
- 60 events during active anomaly investigation
- 25% high-severity events
- High noise + more ambiguous signals
- Elevated interruption rate (20%)

---

## Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Display scrollable event feed with log text | Must |
| FR-02 | Show AI recommendation with confidence category | Must |
| FR-03 | Show evidence traces (cited log lines) | Must (B/C) |
| FR-04 | Record operator action per event | Must |
| FR-05 | Record decision time per event | Must |
| FR-06 | Record action corrections/reversals | Should |
| FR-07 | Filter/search event history | Should |
| FR-08 | Scenario switching (routine/incident) | Must |
| FR-09 | Export interaction telemetry as JSON | Must |
| FR-10 | Display "insufficient evidence" when evidence is weak | Must (B/C) |

## Ergonomic Requirements

| ID | Requirement | Mechanism | Variant |
|----|-------------|-----------|---------|
| ER-01 | Reduce context switching | Evidence in-panel (no separate view) | B, C |
| ER-02 | Prevent high-risk slips | Checklist + confirmation for run_procedure | C |
| ER-03 | Reduce overload in incident scenario | Priority sorting by severity | C |
| ER-04 | Support interruption recovery | "What changed" + "Last action" cues | C |
| ER-05 | Reduce decision paralysis | "Next best step" one-line guidance | C |
| ER-06 | Reduce automation bias | Verifiable evidence reduces blind following | B, C |
| ER-07 | Prevent alert fatigue | Severity badges + visual hierarchy | C |
| ER-08 | Support error recovery | Visual feedback on action submission | All |

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Automation bias (over-trusting AI) | High | High | Evidence panel (B), verification nudges |
| Alert fatigue (ignoring warnings) | Medium | High | Severity prioritization (C) |
| Catastrophic slip (wrong procedure) | Low | Critical | Checklist gate (C) |
| Context loss after interruption | High (fatigued) | Medium | Resumption cues (C) |
| AI hallucination / overconfidence | Medium | High | "Insufficient evidence" refusal (B/C) |
| Checklist fatigue (skipping) | Medium | Medium | Limited to high-risk actions only |

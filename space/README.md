---
title: Fuzzy Abstention
emoji: 🤔
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
tags:
  - calibration
  - abstention
  - ai-safety
  - lexsi-hackathon
---

# Fuzzy Abstention

Is a single confidence score the right representation for deciding when a language model
should abstain? Measured on `CohereLabs/tiny-aya-global` over 210 items, with
out-of-fold probe predictions.

**Headline:** the model's stated confidence is calibrated to **ECE 0.018** where a
question has one clean answer and **0.904** where it does not — while spanning only
0.865–0.955 across cells whose accuracy spans 0%–93.7%.

Dataset: [demiladepy/fuzzy-abstention-items](https://huggingface.co/datasets/demiladepy/fuzzy-abstention-items)

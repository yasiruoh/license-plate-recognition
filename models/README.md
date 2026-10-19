# Model artifacts

The selected pretrained checkpoint is retrieved automatically on first inference and cached by Hugging Face. Internet access and sufficient disk/memory are required for that first download. No local fitting step is required for the default backend.

Plate detection and OCR are separate stages. OCR is configured for English/Latin uppercase characters and digits. Pakistani and other multilingual plates may require a different OCR configuration. No vehicle-owner lookup or tracking is implemented.

Only install trusted artifacts. No private dataset, sample patient image, malicious executable, or fabricated metric is included.

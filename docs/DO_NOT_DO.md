# DO_NOT_DO: Architectural Lessons and Rejected Approaches

### 1. Fallback to ######## Mask
- DO NOT overwrite partially recognized predictions in det.text with ########
  - Reason: In is_seq_match(pred, target), pred=######## only matches when target consists purely of hashes.
    If target has any valid alphanumeric character (e.g. H077OO9# or T213BB#2), pred=######## yields match=False and incurs maximum Levenshtein distance.
  - Rule: Fallback to ######## is allowed strictly when det.text is empty, detector confidence >= 0.80, verifier confirmation p_score >= 0.50, and only for single-line plates (effective_ar >= 1.95). In Type 1A there are no pure hash plates in GT.

### 2. Type 1A Multi-Seam on Confident Predictions
- DO NOT trigger alternative multi-seam offsets when text_1a is already valid GOST without wildcards and confidence >= 0.70.
  - Reason: 2-pixel shifts cut character strokes differently and hallucinate alternative characters (e.g. O000AM80 -> E000AM80, A150AB88 -> A190AB88, A706TT74 -> A708TO190) that overwrite true predictions if raw OCR confidence fluctuates.

### 3. Verifier Thresholds on Stitched Type 1A Crops
- DO NOT apply standard single-line verifier rejection thresholds (p_score < 0.10 or 0.34) to stitched Type 1A crops.
  - Reason: PlateVerifier was trained on natural single-line crops. The central seam and character scale in stitched crops can yield p_score as low as 0.001-0.05 on genuine plates.
  - Rule: For syntax-valid Type 1A plates with ocr_confidence >= 0.65 and det.confidence >= 0.50, only reject if p_score < 0.001.

### 4. CTCDecoder Duplication
- DO NOT duplicate CTCDecoder implementations across ocr.py and decoder.py.
  - Reason: Causes architectural sync drift where decoder updates are ignored by PlateOCR.
  - Rule: Single Source of Truth is src/pipeline/decoder.py.

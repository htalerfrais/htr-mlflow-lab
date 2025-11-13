import Levenshtein


def calculate_cer(reference: str, hypothesis: str) -> float:
    # Normalize: remove spaces and convert to lowercase for HTR evaluation
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")
    
    # Handle edge case: empty reference
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    
    # Calculate CER using Levenshtein distance
    return Levenshtein.distance(ref, hyp) / len(ref)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    returns figure between 0.0 and 1.0
    """
    # Split strings into words (tokens)
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Handle edge case: empty reference
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Calculate WER using Levenshtein distance on word lists
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)

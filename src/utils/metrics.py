import Levenshtein


def calculate_cer(reference: str, hypothesis: str) -> float:
    # Normalize: remove spaces and convert to lowercase for HTR evaluation
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")
    
    # Handle edge case: empty reference
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    
    # Calculate CER using Levenshtein distance
    # cap distance so that we don't get CER > 1
    distance = Levenshtein.distance(ref, hyp)
    capped_distance = min(distance, len(ref))
    return capped_distance / len(ref)


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
    # cap distance so that we don't get WER > 1
    distance = Levenshtein.distance(ref_words, hyp_words)
    capped_distance = min(distance, len(ref_words))
    return capped_distance / len(ref_words)

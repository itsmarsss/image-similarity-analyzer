def combined_score(
    pixel_score: float,
    embed_score: float,
    pose_score: float,
    weights: dict
) -> float:
    """
    Calculate weighted average of the three scores.
    weights = {'pixel': w1, 'embedding': w2, 'pose': w3}
    
    Handles None values by excluding them from calculation and adjusting weights accordingly.
    Returns a score in [0,1] range by normalizing by the sum of active weights.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    # Only include non-None scores in calculation
    if pixel_score is not None and weights['pixel'] > 0:
        weighted_sum += pixel_score * weights['pixel']
        total_weight += weights['pixel']
    
    if embed_score is not None and weights['embedding'] > 0:
        weighted_sum += embed_score * weights['embedding']
        total_weight += weights['embedding']
    
    if pose_score is not None and weights['pose'] > 0:
        weighted_sum += pose_score * weights['pose']
        total_weight += weights['pose']
    
    # Avoid division by zero
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight

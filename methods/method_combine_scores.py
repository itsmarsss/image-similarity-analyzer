def combined_score(
    pixel_score: float,
    embed_score: float,
    pose_score: float,
    weights: dict
) -> float:
    """
    Calculate weighted average of the three scores.
    weights = {'pixel': w1, 'embedding': w2, 'pose': w3}
    
    Returns a score in [0,1] range by normalizing by the sum of weights.
    """
    weighted_sum = (
        pixel_score   * weights['pixel']   +
        embed_score   * weights['embedding'] +
        pose_score    * weights['pose']
    )
    
    total_weight = weights['pixel'] + weights['embedding'] + weights['pose']
    
    # Avoid division by zero
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight

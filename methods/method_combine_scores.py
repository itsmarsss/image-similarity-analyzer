def combined_score(
    pixel_score: float,
    embed_score: float,
    pose_score: float,
    weights: dict
) -> float:
    """
    weights = {'pixel': w1, 'embedding': w2, 'pose': w3}
    """
    return (
        pixel_score   * weights['pixel']   +
        embed_score   * weights['embedding'] +
        pose_score    * weights['pose']
    )

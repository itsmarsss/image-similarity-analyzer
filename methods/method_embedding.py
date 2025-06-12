import numpy as np
import cohere
import cv2
import base64
import os

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embedding_difference_score(
    orig_img: np.ndarray,
    swapped_img: np.ndarray,
    cohere_api_key: str
) -> float:
    """
    Uses Cohere's Embed V4 multimodal model to get embeddings for each image.
    Computes cosine similarity, maps to [0,1] via (1 - sim) / 2.
    """
    client = cohere.ClientV2(api_key=cohere_api_key)
    
    # Save images temporarily
    orig_path = "temp_orig.jpg"
    swapped_path = "temp_swap.jpg"
    cv2.imwrite(orig_path, orig_img)
    cv2.imwrite(swapped_path, swapped_img)
    
    try:
        # Convert images to base64
        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                image_content = image_file.read()
                stringified_buffer = base64.b64encode(image_content).decode("utf-8")
                return f"data:image/jpeg;base64,{stringified_buffer}"
        
        orig_base64 = image_to_base64(orig_path)
        swapped_base64 = image_to_base64(swapped_path)
        
        # Get embeddings using the V2 API
        response = client.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[orig_base64, swapped_base64],
        )
        
        e1, e2 = response.embeddings.float_
        sim = cosine_similarity(np.array(e1), np.array(e2))
        # normalize so that perfect match → 0, opposite → 1
        score = (1 - sim) / 2
        
    finally:
        # Clean up temporary files
        if os.path.exists(orig_path):
            os.remove(orig_path)
        if os.path.exists(swapped_path):
            os.remove(swapped_path)
    
    return score

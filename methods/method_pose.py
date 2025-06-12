import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def pose_difference_score(orig_img: np.ndarray, swapped_img: np.ndarray) -> float:
    """
    Extracts 33 3D landmarks with MediaPipe, normalizes by chest width,
    compares via cosine similarity, maps to [0,1].
    
    Returns:
    - 0.0: Both images have identical poses (perfect match)
    - 0.5: Both images have no detectable poses (neutral)
    - 1.0: One image has pose, other doesn't (maximum difference)
    - (0,1): Calculated similarity when both have poses
    """
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    try:
        # MediaPipe expects RGB:
        o = pose.process(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        s = pose.process(cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB))
        
        orig_has_pose = o.pose_landmarks is not None
        swap_has_pose = s.pose_landmarks is not None
        
        # Case 1: Both images have no detectable pose
        if not orig_has_pose and not swap_has_pose:
            print("Pose detection: Neither image has detectable pose")
            return 0.5  # Neutral score - can't compare poses
        
        # Case 2: Only one image has a detectable pose
        if orig_has_pose != swap_has_pose:
            if orig_has_pose:
                print("Pose detection: Original has pose, swapped doesn't")
            else:
                print("Pose detection: Swapped has pose, original doesn't")
            return 1.0  # Maximum difference
        
        # Case 3: Both images have detectable poses
        print("Pose detection: Both images have detectable poses")
        
        def get_coords(results):
            lm = results.pose_landmarks.landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm])
            # chest left shoulder=11, right shoulder=12
            chest_dist = np.linalg.norm(pts[11] - pts[12])
            if chest_dist == 0:
                print("Warning: Chest distance is zero, using alternative normalization")
                # Use torso height as alternative normalization
                torso_height = abs(pts[11].y - pts[23].y)  # shoulder to hip
                if torso_height > 0:
                    return (pts / torso_height).flatten()
                else:
                    return pts.flatten()  # No normalization if all else fails
            return (pts / chest_dist).flatten()

        v1 = get_coords(o)
        v2 = get_coords(s)
        sim = cosine_similarity(v1, v2)
        score = (1 - sim) / 2
        print(f"Pose similarity: {sim:.4f}, difference score: {score:.4f}")
        return score
        
    finally:
        pose.close()

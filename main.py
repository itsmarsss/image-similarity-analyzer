import argparse
import cv2

from methods.method_pixel_diff import pixel_difference_score
from methods.method_embedding  import embedding_difference_score
from methods.method_pose       import pose_difference_score
from methods.method_combine_scores    import combined_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input',    required=True, help='Path to original image (input.png)')
    p.add_argument('-o', '--output',   required=True, help='Path to self-swap image (output.png)')
    p.add_argument('-w', '--weights',  nargs=3, type=float,
                   default=[1.0, 1.0, 1.0],
                   help='Weights: pixel, embedding, pose')
    p.add_argument('--cohere-key',     required=True,
                   help='Cohere API key')
    args = p.parse_args()

    img1 = cv2.imread(args.input)
    img2 = cv2.imread(args.output)

    ps = pixel_difference_score(img1, img2)
    es = embedding_difference_score(img1, img2, args.cohere_key)
    os = pose_difference_score(img1, img2)

    w = {'pixel':     args.weights[0],
         'embedding': args.weights[1],
         'pose':      args.weights[2]}
    cs = combined_score(ps, es, os, w)

    print(f"Pixel Score:     {ps:.4f}")
    print(f"Embedding Score: {es:.4f}")
    print(f"Pose Score:      {os:.4f}")
    print(f"Combined Score:  {cs:.4f}")

if __name__ == '__main__':
    main()
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--options", type=str, default="train", choices=['train', 'prune', 'test'])

# Path
parser.add_argument("--data_path", type=str, default="/Users/ywj7373/Downloads/faces_emore")
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--ckpt_path", type=str, default="./output/ckpt")
parser.add_argument("--saved_model_path", type=str, default="./output/MobileFaceNet")

# Checkpoints
parser.add_argument("--summary_interval", type=int, default=1000)
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--evaluate_interval", type=int, default=5000)

# Training
parser.add_argument("--epoch_size", type=int, default=5)
parser.add_argument("--class_size", type=int, default=85742)
parser.add_argument("--embedding_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=128, help='batch size for training and evaluation')

# Pruning
parser.add_argument("--filter_size", type=int, default=512)
parser.add_argument("--filter_percentage", type=float, default=0.5)

args = parser.parse_args()
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--options", type=str, default="train", choices=['train', 'prune', 'load'])

# Path
base_path = '/content/drive/MyDrive/data/'
parser.add_argument("--index_path", type=str, default="/content/MFN_Pruning/tran.index")
parser.add_argument("--tfrecord_path", type=str, default="/content/drive/MyDrive/faces_ms1m_112x112/tfrecords/tran.tfrecords")
parser.add_argument("--data_path", type=str, default=base_path)
parser.add_argument("--output_path", type=str, default=base_path + "output")
parser.add_argument("--ckpt_path", type=str, default=base_path + "output/ckpt")
parser.add_argument("--saved_model_path", type=str, default=base_path + "output/MobileFaceNet")
parser.add_argument("--saved_pruned_model_path", type=str, default=base_path + "output/MFN_weights_pruned_3")
parser.add_argument("--pruned_filters_path", type=str, default="/content/MFN_Pruning/pruned_filters.txt")

# Checkpoints
parser.add_argument("--summary_interval", type=int, default=100)
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
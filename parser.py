import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--options", type=str, default="train", choices=['train', 'prune', 'load'])

# Path
base_path = '/content/drive/MyDrive/data/'
tfrecords_path = "/content/drive/MyDrive/faces_ms1m_112x112/tfrecords/"

parser.add_argument("--index_path", type=str, default="/content/MFN_Pruning/tran.index")
parser.add_argument("--tfrecords_file_path", type=str, default=tfrecords_path)
parser.add_argument("--tfrecord_path", type=str, default=tfrecords_path + "tran.tfrecords")
parser.add_argument("--data_path", type=str, default=base_path)
parser.add_argument("--output_path", type=str, default=base_path + "output")
parser.add_argument("--ckpt_path", type=str, default=base_path + "output/ckpt")
parser.add_argument("--pruned_ckpt_path", type=str, default=base_path + "output/pruned_ckpt")
parser.add_argument("--saved_model_path", type=str, default=base_path + "output/ckpt/model_epoch:4_step:225000.pth")
parser.add_argument("--saved_margin_path", type=str, default=base_path + "output/ckpt/head_epoch:4_step:225000.pth")
parser.add_argument("--saved_optimizer_path", type=str, default=base_path + "output/ckpt/optimizer_epoch:4_step:225000.pth")
parser.add_argument("--saved_pruned_model_path", type=str, default=base_path + "output/MFN_weights_pruned_3")
parser.add_argument("--pruned_filters_path", type=str, default=base_path + "output/pruned_filters.txt")

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
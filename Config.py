
import argparse
import torch


# ===================== Argument Parsing =====================
def get_args():
    parser = argparse.ArgumentParser(description="SimCLR")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu") # for runing on GPU
    #parser.add_argument('--device',default="mps" if torch.backends.mps.is_available() else "cpu") # for runing on mps MAC OS

    parser.add_argument('--dataset',  type=str, default="cifar10")
    parser.add_argument('--num_train_samples', type=int, default=10000)
    parser.add_argument('--num_test_samples', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--epochs_eval', type=int, default=40)


    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--debug', type=bool, default=False)


    return parser.parse_args()


args = get_args()




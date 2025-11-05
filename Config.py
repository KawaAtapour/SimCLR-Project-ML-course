
import argparse
import torch


# ===================== Argument Parsing =====================
def get_args():
    parser = argparse.ArgumentParser(description="SimCLR")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu") # for runing on GPU
    #parser.add_argument('--device',default="mps" if torch.backends.mps.is_available() else "cpu") # for runing on mps MAC OS

    parser.add_argument('--dataset',  type=str, default="cifar10")
    parser.add_argument('--num_train_samples', type=int, default=10000)
    parser.add_argument('--num_test_samples', type=int, default=9000)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--epochs_eval', type=int, default=100)


    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--saved_model', type=bool, default=True)
    


    return parser.parse_args()


args = get_args()




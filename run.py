import argparse
import torch
from ptdas.data import DASDataset
from ptdas.models import VAE, CNNVAE
from ptdas.train import train
from ptdas.test import test

def main():
    parser = argparse.ArgumentParser(description="VAE and ConvVAE for DAS data")
    parser.add_argument("--model", type=str, choices=["vae", "cvae"], required=True, help="Choose the model: VAE or ConvVAE")
    parser.add_argument("--mode", type=str, choices=["train", "detect"], required=True, help="Choose mode: train or detect")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation Dataset Percentage")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--n_samples", type=int, default=25600, help="Number of samples to use from the dataset")
    parser.add_argument("--kl_scale", type=float, default=1e-5, help="Scale factor for KL divergence loss")
    parser.add_argument("--data_dir", type=str, default="/cluster/home/jorgenaf/TinyDAS/data", help="Directory containing the training data")
    parser.add_argument("--infer_dir", type=str, default="/cluster/home/jorgenaf/TinyDAS/infer", help="Directory containing the inference data")
    parser.add_argument("--model_path", type=str, help="Path to the saved model for detection mode")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = DASDataset(n=args.n_samples, val_split=args.val_split)

    print(args.batch_size)


    # Choose model
    if args.model.lower() == "vae":
        model = VAE()
    elif args.model.lower() == "cvae":
        model = CNNVAE()

    if args.mode == "train":
        train(model, dataset, args.num_epochs, args.batch_size, args.learning_rate, device, args.kl_scale)
    elif args.mode == "detect":
        if args.model_path is None:
            raise ValueError("Model path must be provided for detection mode")
        
        # Load the trained model
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        # Test
        #test(model, args.infer_dir, device)

if __name__ == '__main__':
    main()
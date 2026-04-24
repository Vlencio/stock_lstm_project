from torch.utils.data import DataLoader
from dataset import StockDataset, create_time_series_splits
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='Create DataLoaders for stock dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Parquet files.')
    parser.add_argument('--window', type=int, default=30, help='Window size for time series data.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data for training.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of data for validation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')

    args = parser.parse_args()

    # Load all parquet files from the specified directory
    parquer_files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))

    # Create dataset
    dataset = StockDataset(data_source=parquer_files, window=args.window)

    # split dataset into train, val, test
    splits = create_time_series_splits(
        dataset,
        train_ratio = args.train_ratio,
        val_ratio = args.val_ratio
        )
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(splits['val'], batch_size=args.batch_size, shuffle=False),
        'test': DataLoader(splits['test'], batch_size=args.batch_size, shuffle=False)
    }

    # Display split informations 
    print(f"Number of training samples: {len(splits['train'])}")
    print(f"Number of validation samples: {len(splits['val'])}")
    print(f"Number of test samples: {len(splits['test'])}")

    # Example: Iterate through one batch of training data
    for split_name, loadr in dataloaders.items():
        batch_x, batch_y = next(iter(loadr))
        print(f"\n{split_name.capitalize()} Batch shapes: X = {batch_x.shape}, Y =: {batch_y.shape}")

if __name__ == '__main__':
    main()
# scripts/train.py (updated with normalization)
"""
Training script for LSTM stock prediction model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import glob
import joblib
import sys
from pathlib import Path
from colorama import Fore, Style
from tqdm import tqdm

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import local modules
from scripts.model import StockLSTM
from scripts.dataset import create_datasets_with_scaler
from utils.feature_engineering import FEATURE_COLS, TARGET_COL
from utils.logger import TrainingLogger
from utils.reporter import TrainingReporter
from utils.progress import ColoredProgress, TrainingProgressBar, ValidationProgressBar
from utils.visualizer import TrainingVisualizer

class Trainer:
    """Main trainer class"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, checkpoint_dir, scaler, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scaler = scaler  # Store scaler for later use
        self.args = args
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger and reporter
        self.logger = TrainingLogger(
            log_dir=str(log_dir),
            experiment_name=args.experiment_name
        )
        self.reporter = TrainingReporter(str(log_dir))
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Save hyperparameters and scaler info
        hyperparams = vars(args)
        hyperparams['scaler_params'] = self.scaler.get_params()
        self.logger.save_hyperparameters(hyperparams)
        
        # Save scaler
        scaler_path = self.checkpoint_dir / 'scaler.pkl'
        self.scaler.save(scaler_path)
        ColoredProgress.print_success(f"Scaler saved to {scaler_path}")

        # Also save with joblib for use by backtest/paper_trade
        joblib_path = self.checkpoint_dir / 'scaler_joblib.pkl'
        joblib.dump(scaler.scaler, str(joblib_path))
        ColoredProgress.print_success(f"Joblib scaler saved to {joblib_path}")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'args': vars(self.args),
            'scaler_params': self.scaler.get_params()
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            ColoredProgress.print_success(f"Saved best model (loss: {val_loss:.6f})")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        
        # Log epoch start
        self.logger.log_epoch_start(epoch, self.args.epochs)
        
        # Calculate logging interval (25% of epoch)
        log_interval = max(1, len(self.train_loader) // 4)
        
        epoch_start_time = time.time()
        
        # Create progress bar
        progress = TrainingProgressBar(self.train_loader, epoch, self.args.epochs)
        
        with progress as pbar:
            for batch_idx, (data, target) in pbar:
                step_start_time = time.time()
                
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Reshape if needed
                if len(data.shape) == 2:
                    data = data.unsqueeze(-1)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                step_time = time.time() - step_start_time
                
                # Update progress bar
                progress.update_metrics(pbar, batch_idx, {
                    'loss': avg_loss,
                    'step_time': step_time
                })
                
                # Log at 25% intervals
                if (batch_idx + 1) % log_interval == 0 or batch_idx == len(self.train_loader) - 1:
                    progress_pct = ((batch_idx + 1) / len(self.train_loader)) * 100
                    
                    self.logger.log_training_metrics(
                        epoch=epoch,
                        step=batch_idx + 1,
                        total_steps=len(self.train_loader),
                        metrics={
                            'train_loss': avg_loss,
                            'step_time_ms': step_time * 1000,
                            'iterations_per_second': 1 / step_time
                        }
                    )
                    
                    ColoredProgress.print_info(
                        f"[{progress_pct:.0f}% Complete] Loss: {avg_loss:.6f}"
                    )
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = running_loss / len(self.train_loader)
        
        ColoredProgress.print_success(
            f"Epoch {epoch} completed in {ColoredProgress.format_time(epoch_time)}"
        )
        
        return avg_epoch_loss, epoch_time
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        
        # Simple progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"{Fore.CYAN}Validating{Style.RESET_ALL}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if len(data.shape) == 2:
                    data = data.unsqueeze(-1)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                val_loss += loss.item()
        
        pbar.close()
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Log validation metrics
        self.logger.log_validation_metrics(epoch, {'val_loss': avg_val_loss})
        
        print(f"Validation Loss: {ColoredProgress.format_loss(avg_val_loss)}\n")
        
        return avg_val_loss
    
    def train(self):
        """Main training loop"""
        ColoredProgress.print_header("Starting Training")
        
        print(f"Device: {ColoredProgress.format_metric(self.device)}")
        print(f"Epochs: {ColoredProgress.format_metric(self.args.epochs)}")
        print(f"Batch Size: {ColoredProgress.format_metric(self.args.batch_size)}")
        print(f"Learning Rate: {ColoredProgress.format_metric(self.args.lr)}")
        print(f"Scaler Type: {ColoredProgress.format_metric(self.args.scaler_type)}\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Log epoch end
            self.logger.log_epoch_end(epoch, train_loss, val_loss, epoch_time)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Add epoch report
            self.reporter.add_epoch_report(epoch, train_loss, val_loss, self.best_val_loss, epoch_time)
        
        # Save final summary
        self.reporter.save_summary()
        self.reporter.print_summary()
        
        # Create visualizations
        ColoredProgress.print_info("Generating training visualizations...")
        visualizer = TrainingVisualizer(save_dir=self.args.log_dir)
        visualizer.plot_losses(
            self.train_losses,
            self.val_losses,
            title=f'{self.args.experiment_name} - Training History'
        )

        ColoredProgress.print_header("Training Completed!")
        ColoredProgress.print_success(f"Best Validation Loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for Stock Prediction')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed parquet files (output of collect_data.py)')
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Ticker symbol — must match a <SYMBOL>.parquet file in data_dir')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--scaler_type', type=str, default='minmax', choices=['minmax', 'standard'])

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Save directories
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--experiment_name', type=str, default='stock_lstm')

    args = parser.parse_args()

    # Resolve data file — one parquet per symbol in processed dir
    parquet_files = glob.glob(f"{args.data_dir}/{args.symbol}.parquet")
    if not parquet_files:
        # Fallback: load all parquets if no symbol-specific file found
        parquet_files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {args.data_dir}. "
            f"Run collect_data.py first to generate processed data."
        )

    ColoredProgress.print_info(f"Loading data from: {parquet_files}")

    data_dict = create_datasets_with_scaler(
        parquet_files,
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        scaler_type=args.scaler_type,
    )

    train_dataset = data_dict['train']
    val_dataset = data_dict['val']
    scaler = data_dict['scaler']

    ColoredProgress.print_success(f"Data loaded using {args.scaler_type} scaler")
    ColoredProgress.print_info(f"Features: {len(FEATURE_COLS)} columns")
    ColoredProgress.print_info(
        f"Train / Val / Test: {len(train_dataset)} / {len(val_dataset)} / {len(data_dict['test'])}\n"
    )

    # DataLoaders — training shuffles batch order (not sequence order), val is sequential
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model — input_size matches number of features (dynamic, not hardcoded)
    input_size = len(FEATURE_COLS)
    model = StockLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=1,
    ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        scaler=scaler,
        args=args,
    )

    trainer.train()


if __name__ == '__main__':
    main()

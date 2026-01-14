"""Training script for Vietnamese NER model."""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import get_config
from data.datasets import DataCollator, NERDataset
from data.preprocessors import CoNLLPreprocessor
from models.phobert_ner import PhoBERTForNER, PhoBERTWithCRF
from training.trainer import NERTrainer
from utils.helpers import set_seed
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Vietnamese NER model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="phobert",
        choices=["phobert", "phobert_crf"],
        help="Model type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup logger
    logger = setup_logger(
        "train",
        log_file=os.path.join(config.log_dir, "train.log")
    )
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model
    )
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(config.data.label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Load datasets
    logger.info("Loading datasets...")
    preprocessor = CoNLLPreprocessor()
    
    train_sentences = preprocessor.read_conll_file(config.data.train_path)
    eval_sentences = preprocessor.read_conll_file(config.data.valid_path)
    
    logger.info(f"Train samples: {len(train_sentences)}")
    logger.info(f"Eval samples: {len(eval_sentences)}")
    
    # Create datasets
    train_dataset = NERDataset(
        train_sentences,
        tokenizer,
        label2id,
        config.model.max_seq_length
    )
    
    eval_dataset = NERDataset(
        eval_sentences,
        tokenizer,
        label2id,
        config.model.max_seq_length
    )
    
    # Create data loaders
    data_collator = DataCollator()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    
    if args.model_type == "phobert":
        model = PhoBERTForNER(
            model_name=config.model.pretrained_model,
            num_labels=config.model.num_labels,
            dropout=config.model.dropout,
            hidden_size=config.model.hidden_size
        )
    else:  # phobert_crf
        model = PhoBERTWithCRF(
            model_name=config.model.pretrained_model,
            num_labels=config.model.num_labels,
            dropout=config.model.dropout,
            hidden_size=config.model.hidden_size
        )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(
        f"Trainable parameters: {model.get_trainable_parameters():,}"
    )
    
    # Create trainer
    trainer = NERTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=config.output_dir,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        max_grad_norm=config.training.max_grad_norm,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Training loop
    logger.info("Starting training loop...")
    
    for epoch in range(config.training.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        logger.info(f"{'='*50}")
        
        train_loss = trainer.train_epoch(epoch + 1)
        logger.info(f"Average train loss: {train_loss:.4f}")
    
    # Final evaluation
    logger.info("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final metrics: {final_metrics}")
    
    # Save final model
    trainer.save_model("final_model.pt")
    
    logger.info("\nTraining completed!")
    logger.info(f"Best F1 score: {trainer.best_f1:.4f}")
    logger.info(f"Models saved to {config.output_dir}")


if __name__ == "__main__":
    main()
import argparse
import logging
import os
from typing import List, Optional

import torch
from .model.transformer import ShiemiTransformer
from .tokenizer.tokenizer import ShiemiTokenizer
from .tokenizer.train_tokenizer import TokenizerTrainer
from .data.dataset import AnimeTextDataset
from .training.trainer import ShiemiTrainer
from .generation.generator import TextGenerator
from .config.model_config import ShiemiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tokenizer(args):
    """Train the SentencePiece tokenizer."""
    trainer = TokenizerTrainer(vocab_size=args.vocab_size)
    trainer.train(
        input_files=args.input_files,
        output_dir=args.output_dir,
        model_prefix="shiemi",
        input_sentence_size=args.input_sentence_size
    )
    logger.info(f"Tokenizer trained and saved to {args.output_dir}")


def train_model(args):
    """Train the Shiemi model."""
    # Load tokenizer
    tokenizer = ShiemiTokenizer(args.tokenizer_path)

    # Create model
    config = ShiemiConfig()
    model = ShiemiTransformer(config)

    # Load datasets
    train_dataset = AnimeTextDataset(
        file_paths=args.train_files,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )

    val_dataset = None
    if args.val_files:
        val_dataset = AnimeTextDataset(
            file_paths=args.val_files,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
        )

    # Extract training arguments
    training_args = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size
    }

    # Create trainer
    trainer = ShiemiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        **training_args
    )

    # Train
    trainer.train()
    logger.info(f"Model training completed. Checkpoints saved to {
                args.output_dir}")


def chat(args):
    """Chat with Shiemi."""
    # Load model and tokenizer
    config = ShiemiConfig()
    model = ShiemiTransformer(config)

    # Use safe_globals context manager to allow loading ShiemiConfig
    with torch.serialization.safe_globals([ShiemiConfig]):
        checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = ShiemiTokenizer(args.tokenizer_path)
    generator = TextGenerator(model, tokenizer)

    # Start chat loop
    print("\nWelcome to Shiemi! Type 'quit' or 'exit' to end the conversation.\n")
    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        response = generator.chat(
            user_input,
            chat_history=chat_history,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )

        print(f"Shiemi: {response}\n")

        chat_history.append({"user": user_input, "assistant": response})
        if len(chat_history) > 10:  # Keep chat history manageable
            chat_history = chat_history[-10:]


def main():
    parser = argparse.ArgumentParser(description="Shiemi CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Tokenizer training
    tokenizer_parser = subparsers.add_parser(
        "train-tokenizer", help="Train the tokenizer")
    tokenizer_parser.add_argument(
        "--input_files", nargs="+", required=True, help="Input files for training")
    tokenizer_parser.add_argument(
        "--output_dir", required=True, help="Output directory")
    tokenizer_parser.add_argument(
        "--vocab_size", type=int, default=32000, help="Vocabulary size")
    tokenizer_parser.add_argument(
        "--input_sentence_size", type=int, help="Number of training sentences")

    # Model training
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--train_files", nargs="+", required=True, help="Training data files")
    train_parser.add_argument("--val_files", nargs="+",
                              help="Validation data files")
    train_parser.add_argument(
        "--tokenizer_path", required=True, help="Path to tokenizer model")
    train_parser.add_argument(
        "--output_dir", required=True, help="Output directory for checkpoints")
    train_parser.add_argument("--num_epochs", type=int,
                              default=10, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int,
                              default=32, help="Batch size")
    train_parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Chat with Shiemi")
    chat_parser.add_argument(
        "--model_path", required=True, help="Path to model checkpoint")
    chat_parser.add_argument(
        "--tokenizer_path", required=True, help="Path to tokenizer model")
    chat_parser.add_argument("--max_length", type=int,
                             default=200, help="Maximum response length")
    chat_parser.add_argument("--temperature", type=float,
                             default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--top_p", type=float,
                             default=0.9, help="Nucleus sampling probability")

    args = parser.parse_args()

    if args.command == "train-tokenizer":
        train_tokenizer(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "chat":
        chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

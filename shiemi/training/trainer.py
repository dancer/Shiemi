import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import os
from ..model.transformer import ShiemiTransformer
from ..data.dataset import AnimeTextDataset

logger = logging.getLogger(__name__)


class ShiemiTrainer:
    def __init__(
        self,
        model: ShiemiTransformer,
        train_dataset: AnimeTextDataset,
        val_dataset: Optional[AnimeTextDataset] = None,
        output_dir: str = "checkpoints",
        **training_args
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Training arguments
        self.training_args = {
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "num_epochs": 10,
            "batch_size": 32,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 2000,
            "log_every": 10,
            "save_every": 1000,
            "eval_every": 1000,
            **training_args
        }

        # Setup optimizer
        self.optimizer = model.configure_optimizers(
            self.training_args["weight_decay"],
            self.training_args["learning_rate"]
        )

        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
            collate_fn=AnimeTextDataset.collate_fn,
            num_workers=0,  # Reduced for stability
            pin_memory=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_args["batch_size"],
                shuffle=False,
                collate_fn=AnimeTextDataset.collate_fn,
                num_workers=0,  # Reduced for stability
                pin_memory=True
            )

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(
        ) if model.config.use_mixed_precision else None

    def train(self):
        """Main training loop."""
        device = torch.device("cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.model.to(device)

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(self.training_args["num_epochs"]):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )

                    # Calculate loss
                    loss = F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        batch["labels"].view(-1),
                        ignore_index=-100
                    )
                    loss = loss / \
                        self.training_args["gradient_accumulation_steps"]

                # Backward pass with mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % self.training_args["gradient_accumulation_steps"] == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_args["max_grad_norm"])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_args["max_grad_norm"])
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.training_args["log_every"] == 0:
                        logger.info(f"Step {global_step}: loss = {
                                    loss.item() * self.training_args['gradient_accumulation_steps']:.4f}")

                    # Save checkpoint
                    if global_step % self.training_args["save_every"] == 0:
                        self.save_checkpoint(f"step_{global_step}")

                    # Evaluation
                    if self.val_dataset and global_step % self.training_args["eval_every"] == 0:
                        val_loss = self.evaluate()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint("best")

                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training completed. Final checkpoint saved.")

    def evaluate(self) -> float:
        """Evaluate the model on validation set."""
        device = next(self.model.parameters()).device
        self.model.eval()
        total_loss = 0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps
        logger.info(f"Validation loss: {avg_loss:.4f}")
        self.model.train()
        return avg_loss

    def save_checkpoint(self, name: str):
        """Save a checkpoint of the model."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "training_args": self.training_args
        }
        path = os.path.join(self.output_dir, f"checkpoint_{name}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, model: ShiemiTransformer, **kwargs) -> "ShiemiTrainer":
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(model, **kwargs)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return trainer

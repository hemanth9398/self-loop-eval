"""LoRA fine-tuning trainer — updates the student model using PEFT."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from self_loop_eval.config import TrainingConfig

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA fine-tuner for the student model.

    Uses HuggingFace PEFT to apply LoRA adapters and train on SFT pairs
    generated from the self-eval loop. Requires the 'training' optional
    dependencies (torch, transformers, peft, etc.).
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(
        self,
        model_name_or_path: str,
        sft_data_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Fine-tune a model using LoRA on SFT training data.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            sft_data_path: Path to JSON file with SFT pairs.
            output_dir: Where to save the fine-tuned adapter. Defaults to config.

        Returns:
            Path to the saved LoRA adapter.

        Raises:
            ImportError: If training dependencies are not installed.
        """
        try:
            from datasets import Dataset
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            raise ImportError(
                "Training dependencies not installed. "
                "Install with: pip install self-loop-eval[training]"
            ) from e

        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load SFT data
        sft_data = self._load_sft_data(sft_data_path)
        if not sft_data:
            logger.warning("No training data found at %s", sft_data_path)
            return output_dir

        logger.info("Loading model: %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            load_in_8bit=True,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare dataset
        dataset = self._prepare_dataset(sft_data, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=10,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Starting LoRA fine-tuning...")
        trainer.train()

        # Save adapter
        adapter_path = output_dir / "lora_adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        logger.info("LoRA adapter saved to %s", adapter_path)
        return adapter_path

    def _prepare_dataset(self, sft_data: list[dict], tokenizer) -> "Dataset":
        """Prepare a HuggingFace Dataset from SFT pairs."""
        from datasets import Dataset

        texts = []
        for pair in sft_data:
            text = (
                f"### Instruction:\n{pair['instruction']}\n\n"
                f"### Response:\n{pair['response']}"
            )
            texts.append(text)

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        return dataset

    @staticmethod
    def _load_sft_data(path: str | Path) -> list[dict]:
        """Load SFT data from a JSON file."""
        path = Path(path)
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

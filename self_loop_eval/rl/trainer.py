"""RL trainer — PPO/DPO training using self-eval rewards."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from self_loop_eval.config import RLConfig, TrainingConfig
from self_loop_eval.eval_loop.round_state import LoopResult
from self_loop_eval.rl.rewards import RewardFunction

logger = logging.getLogger(__name__)


class RLTrainer:
    """RL trainer that uses PPO or DPO on top of self-eval rewards.

    For DPO, the 'chosen' response is the improved solution (higher reward)
    and the 'rejected' response is the original solution (lower reward).

    For PPO, the reward signal from RewardFunction is used directly.

    Requires the 'training' optional dependencies.
    """

    def __init__(self, rl_config: RLConfig, training_config: TrainingConfig):
        self.rl_config = rl_config
        self.training_config = training_config
        self.reward_fn = RewardFunction(rl_config)

    def prepare_dpo_data(
        self, loop_results: list[LoopResult]
    ) -> list[dict]:
        """Prepare DPO preference pairs from loop results.

        Each pair consists of:
        - prompt: the task
        - chosen: the better solution (later round / higher score)
        - rejected: the worse solution (earlier round / lower score)

        Args:
            loop_results: Completed loop results.

        Returns:
            List of DPO-formatted preference pairs.
        """
        pairs = []

        for result in loop_results:
            if len(result.rounds) < 2:
                continue

            first = result.rounds[0]
            last = result.rounds[-1]

            # Only create a pair if there's actual improvement
            first_score = first.env_score or 0.0
            last_score = last.env_score or 0.0

            if last_score > first_score:
                pairs.append({
                    "prompt": result.task_prompt,
                    "chosen": last.solution.content,
                    "rejected": first.solution.content,
                    "chosen_score": last_score,
                    "rejected_score": first_score,
                    "task_id": result.task_id,
                })

        logger.info("Prepared %d DPO preference pairs", len(pairs))
        return pairs

    def train_dpo(
        self,
        model_name_or_path: str,
        loop_results: list[LoopResult],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Run DPO training on the student model.

        Args:
            model_name_or_path: Model to train.
            loop_results: Completed loop results for training data.
            output_dir: Output directory for the trained model.

        Returns:
            Path to the trained model.

        Raises:
            ImportError: If training dependencies not installed.
        """
        try:
            from datasets import Dataset
            from peft import LoraConfig, TaskType
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import DPOConfig, DPOTrainer
        except ImportError as e:
            raise ImportError(
                "RL training dependencies not installed. "
                "Install with: pip install self-loop-eval[training]"
            ) from e

        output_dir = Path(output_dir or self.training_config.output_dir) / "dpo"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        dpo_pairs = self.prepare_dpo_data(loop_results)
        if not dpo_pairs:
            logger.warning("No DPO pairs available for training")
            return output_dir

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )

        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )

        # Dataset
        dataset = Dataset.from_list(dpo_pairs)

        # DPO config
        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.rl_config.ppo_batch_size,
            learning_rate=self.training_config.learning_rate,
            beta=self.rl_config.dpo_beta,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        logger.info("Starting DPO training with %d pairs", len(dpo_pairs))
        trainer.train()

        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))

        logger.info("DPO training complete. Model saved to %s", output_dir / "final")
        return output_dir / "final"

    def save_dpo_data(
        self, loop_results: list[LoopResult], output_path: str | Path
    ) -> Path:
        """Save DPO data to a JSON file (without training).

        Args:
            loop_results: Completed loop results.
            output_path: File path to save.

        Returns:
            Path to the saved file.
        """
        pairs = self.prepare_dpo_data(loop_results)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(pairs, f, indent=2)

        logger.info("Saved %d DPO pairs to %s", len(pairs), output_path)
        return output_path

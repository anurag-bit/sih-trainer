import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import HfApi, HfFolder, Repository
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    else:
        import gc
        gc.collect()
        logger.info("System memory freed")

def train():
    try:
        hf_api_key = "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"
        model_name = "google/gemma-2b-it"

        logger.info("Loading model")
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key, attn_implementation='eager')
        model.config.use_cache = False

        if torch.cuda.is_available():
            logger.info(f"Moving model to GPU")
            model = model.to('cuda')
        else:
            logger.warning("No GPU detected. Training will proceed on CPU, which may be very slow.")

        logger.info("Loading dataset")
        dataset = load_dataset("prof-freakenstein/sihFinal")

        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)

        def preprocess_function(examples):
            model_inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        logger.info("Preprocessing dataset")
        tokenized_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=1 if torch.cuda.is_available() else 1,
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),
            logging_dir='./logs',
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Reduced for CPU training
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=5e-5,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=500,
            report_to="tensorboard",
            load_best_model_at_end=True,
        )

        logger.info("Creating Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=lambda data: dict((key, torch.stack([d[key] for d in data])) for key in data[0]),
        )

        logger.info("Starting training")
        free_memory()
        trainer.train(resume_from_checkpoint=True)

        logger.info("Saving model")
        model_dir = "./your_trained_model"
        trainer.save_model(model_dir)
        upload_model_to_hub(model_dir, "PHISIH", "PHISIH/PHISIH", hf_api_key)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())

def upload_model_to_hub(model_dir, repo_name, model_id, hf_api_key):
    try:
        HfFolder.save_token(hf_api_key)
        api = HfApi()
        api.create_repo(repo_id=model_id, private=False)
        logger.info(f"Repository '{repo_name}' created successfully.")

        repo = Repository(local_dir=model_dir, clone_from=model_id, use_auth_token=hf_api_key)
        repo.push_to_hub(commit_message="Initial commit")
        logger.info("Model uploaded to Hugging Face Hub")
    except Exception as e:
        logger.error(f"Error uploading model to Hugging Face Hub: {e}")

if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Starting training on {num_gpus} GPU(s)")
        else:
            logger.warning("No GPUs detected. Starting training on CPU.")
        train()
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())

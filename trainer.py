import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import HfApi, HfFolder, Repository
import os
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        logger.info(f"Process group initialized for rank {rank}")
    except Exception as e:
        logger.error(f"Error initializing process group for rank {rank}: {e}")
        raise

def cleanup():
    try:
        dist.destroy_process_group()
        logger.info("Process group destroyed")
    except Exception as e:
        logger.error(f"Error destroying process group: {e}")

def free_gpu_cache():
    try:
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")

def train(rank, world_size):
    try:
        setup(rank, world_size)

        hf_api_key = "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"
        model_name = "google/gemma-2b-it"

        logger.info(f"Rank {rank}: Loading model")
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key, attn_implementation='eager')
        model.config.use_cache = False
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])

        logger.info(f"Rank {rank}: Loading dataset")
        dataset = load_dataset("prof-freakenstein/sihFinal")

        logger.info(f"Rank {rank}: Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)

        def preprocess_function(examples):
            model_inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        logger.info(f"Rank {rank}: Preprocessing dataset")
        tokenized_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

        train_sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(tokenized_dataset, sampler=train_sampler, batch_size=4)

        logger.info(f"Rank {rank}: Setting up training arguments")
        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            fp16=True,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=4,
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

        logger.info(f"Rank {rank}: Creating Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=lambda data: dict((key, torch.stack([d[key] for d in data])) for key in data[0]),
        )

        logger.info(f"Rank {rank}: Starting training")
        free_gpu_cache()
        trainer.train(resume_from_checkpoint=True)

        if rank == 0:
            logger.info("Saving model")
            model_dir = "./your_trained_model"
            trainer.save_model(model_dir)
            upload_model_to_hub(model_dir, "PHISIH", "PHISIH/PHISIH", hf_api_key)

    except Exception as e:
        logger.error(f"Rank {rank}: Error during training: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup()

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
        world_size = torch.cuda.device_count()
        logger.info(f"Starting distributed training on {world_size} GPUs")
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())

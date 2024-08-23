import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import HfApi, HfFolder, Repository
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def free_gpu_cache():
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()
    print("GPU cache cleared.")

def train(rank, world_size):
    setup(rank, world_size)

    hf_api_key = "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"
    model_name = "google/gemma-2b-it"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key, attn_implementation='eager')
    model.config.use_cache = False
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load dataset
    dataset = load_dataset("prof-freakenstein/sihFinal")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    # Create DataLoader with DistributedSampler
    train_sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(tokenized_dataset, sampler=train_sampler, batch_size=4)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10,
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
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: dict((key, torch.stack([d[key] for d in data])) for key in data[0]),
    )

    # Training
    free_gpu_cache()
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")
        free_gpu_cache()
        raise

    # Save model (only on main process)
    if rank == 0:
        model_dir = "./your_trained_model"
        trainer.save_model(model_dir)
        upload_model_to_hub(model_dir, "PHISIH", "PHISIH/PHISIH", hf_api_key)

    cleanup()

def upload_model_to_hub(model_dir, repo_name, model_id, hf_api_key):
    HfFolder.save_token(hf_api_key)
    api = HfApi()
    try:
        api.create_repo(repo_id=model_id, private=False)
        print(f"Repository '{repo_name}' created successfully.")
    except Exception as e:
        print(f"Repository creation failed: {e}")

    repo = Repository(local_dir=model_dir, clone_from=model_id, use_auth_token=hf_api_key)
    repo.push_to_hub(commit_message="Initial commit")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

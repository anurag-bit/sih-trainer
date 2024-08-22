import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import HfApi, HfFolder, Repository

# Function to clear GPU cache
def free_gpu_cache():
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()
    print("GPU cache cleared.")

# Replace with your actual API key
hf_api_key = "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"

# Load the Phi-3 model with eager attention implementation
model_name = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key, attn_implementation='eager')

# Set use_cache to False if needed
model.config.use_cache = False

# Load the tokenized dataset
tokenized_dataset = load_from_disk("/mnt/data/tokenized_dataset")

# Define training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Further reduced batch size
    gradient_accumulation_steps=8,
    fp16=True,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
)

# Enable gradient checkpointing if available
if torch.cuda.is_available():
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Load the train dataset
)

# Clear GPU cache before starting training
free_gpu_cache()

# Start training
try:
    trainer.train()
except RuntimeError as e:
    print(f"RuntimeError during training: {e}")
    free_gpu_cache()
    raise

# Save the trained model
model_dir = "./your_trained_model"
trainer.save_model(model_dir)

# Define repository details
repo_name = "PHISIH"
model_id = f"{repo_name}/{repo_name}"

# Function to upload the model to the Hugging Face Hub
def upload_model_to_hub(model_dir, repo_name, model_id, hf_api_key):
    # Authenticate
    HfFolder.save_token(hf_api_key)
    api = HfApi()

    # Create a new repository
    try:
        api.create_repo(repo_id=model_id, private=False)
        print(f"Repository '{repo_name}' created successfully.")
    except Exception as e:
        print(f"Repository creation failed: {e}")

    # Clone the repository
    repo = Repository(local_dir=model_dir, clone_from=model_id, use_auth_token=hf_api_key)

    # Add files to the repository
    repo.push_to_hub(commit_message="Initial commit")

# Upload the model
try:
    upload_model_to_hub(model_dir, repo_name, model_id, hf_api_key)
except Exception as e:
    print(f"Error uploading model to Hugging Face Hub: {e}")

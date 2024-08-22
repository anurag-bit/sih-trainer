import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import HfApi, HfFolder, Repository

# Function to clear GPU cache
def free_gpu_cache():
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()
    print("GPU cache cleared.")

# Replace with your actual API key
hf_api_key = "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"

# Load the Gemma 2B model
model_name = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key, attn_implementation='eager')

# Set use_cache to False if needed
model.config.use_cache = False

# Load the dataset from Hugging Face
dataset = load_dataset("prof-freakenstein/sihFinal")
# Load the dataset from Hugging Face
dataset = load_dataset("prof-freakenstein/sihFinal")

# Print dataset information
print("Dataset structure:")
print(dataset)

# Print the first few examples
print("\nFirst few examples:")
for i, example in enumerate(dataset['train'][:5]):
    print(f"Example {i + 1}:")
    print(example)
    print()

# Print column names
print("Column names:")
print(dataset['train'].column_names)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)

# Preprocess the dataset (tokenization)
def preprocess_function(examples):
    # Modify this function based on the actual structure of your dataset
    inputs = [f"Human: {example['text'].split('Human: ')[-1]}" for example in examples['text']]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)

# Preprocess the dataset (tokenization)
def preprocess_function(examples):
    # Combine instruction and response for causal language modeling
    inputs = [f"Human: {instr}\n\nAssistant: {resp}" for instr, resp in zip(examples['instruction'], examples['response'])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# Define training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    remove_unused_columns=False,
)

# Enable gradient checkpointing if available
if torch.cuda.is_available():
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
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

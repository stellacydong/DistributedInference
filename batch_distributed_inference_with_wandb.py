# import wandb
# from accelerate import Accelerator
# from accelerate.utils import gather_object
# from datasets import load_dataset
# import torch
# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, TrainerControl, TrainerCallback, TrainerState

import wandb
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import gather_object
from datasets import load_dataset
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, TrainerControl, TrainerCallback, TrainerState


# wandb.init(sync_tensorboard=True)


# Initialize Weights & Biases
wandb.init(project="batch_TinyMistral", config={"batch_size": 16, "epochs": 4})

# Initialize Accelerator
accelerator = Accelerator()

# Define a callback to log GPU metrics with Weights & Biases
class WandbCallback(TrainerCallback):
    def __init__(self):
        self.step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log some metrics (e.g., GPU memory usage)
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        gpu_free_mem = torch.cuda.memory_reserved() / (1024 ** 3)
        wandb.log({"GPU Memory Usage (GB)": gpu_mem, "Free GPU Memory (GB)": gpu_free_mem}, step=self.step)
        self.step += 1

# Define the training function
def hello_world():
    # Load model and tokenizer
    model_path = "Locutusque/TinyMistral-248M"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load and tokenize dataset
    dataset = load_dataset("timdettmers/openassistant-guanaco")

    # Tokenization function
    def tokenize(element):
        return tokenizer(element["text"], truncation=True, max_length=512, add_special_tokens=False)

    dataset_tokenized = dataset.map(tokenize, batched=True, num_proc=os.cpu_count(), remove_columns=["text"])

    # Collate function
    def collate(elements):
        tokenlist = [e["input_ids"] for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])
        input_ids, labels, attention_masks = [], [], []

        for tokens in tokenlist:
            pad_len = tokens_maxlen - len(tokens)
            input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
            labels.append(tokens + [-100] * pad_len)
            attention_masks.append([1] * len(tokens) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
        }

    # List of prompts for evaluation
    prompt_template = "### Human: {}\n### Assistant:"
    questions = [
        "Hello! Who are you? Introduce yourself, please",
        "How much is 2 + 2? Think step-by-step",
        "What is on your mind?",
        "Define artificial general intelligence",
    ] * 10

    prompts = [prompt_template.format(q) for q in questions]

    # Callback class for generation during training
    class GenerateEvalCallback(TrainerCallback):
        def __init__(self, prompts, accelerator):
            self.prompts_all = prompts
            self.accelerator = accelerator

        def prepare_prompts(self, prompts, tokenizer):
            tokenizer.padding_side = "left"
            prompts_tok = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=False, pad_to_multiple_of=8, add_special_tokens=False).to("cuda")
            tokenizer.padding_side = "right"
            return prompts_tok

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, tokenizer, eval_dataloader, **kwargs):
            model.eval()
            model.config.use_cache = True

            # Split questions among GPUs
            with accelerator.split_between_processes(self.prompts_all) as prompts:
                # Batched inference on each GPU
                bs = 8
                # batches = [prompts[i:i + bs] for i in range (0, len(prompts)), bs]
                batches = [prompts[i:i + bs] for i in range(0, len(prompts), bs)]
                outputs = []

                for prompt_batch in batches:
                    prompts_tok = self.prepare_prompts(prompt_batch, tokenizer)
                    with torch.no_grad():
                        outputs_tok = model.generate(**prompts_tok, max_new_tokens=30).to("cpu")
                    outputs.extend([
                        tokenizer.decode(outputs_tok[i][outputs_tok[i] != tokenizer.pad_token_id])
                        for i in range(0, len(outputs_tok))
                    ])

            outputs_gathered = gather_object(outputs)

            # Print a few to console
            accelerator.print(f"EPOCH {state.epoch:0.2f}:")
            for output in outputs_gathered[:5]:
                accelerator.print(output)

            # Write all to file
            if accelerator.is_main_process:
                with open(f"outputs_epoch-{state.epoch:0.2f}.txt", "w") as file:
                    file.write("\n\n".join(outputs_gathered))

            model.config.use_cache = False
            return control

    # Set TrainingArguments and Trainer
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        logging_steps=1,
        num_train_epochs=4,
        learning_rate=0.001,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=collate,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
    )

    # Add callbacks
    trainer.add_callback(
        GenerateEvalCallback(prompts, accelerator)
    )

    # Add Weights & Biases callback for GPU monitoring
    trainer.add_callback(WandbCallback())

    trainer.train()

# Launch the notebook with specified number of processes
notebook_launcher(hello_world, num_processes=8)

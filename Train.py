import torch
import torch.nn as nn
import math
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tokenizer
import os
tokenizer.load_tokenizer() # load saved tokenizer

merges   = tokenizer.merges # bpe merge rules
word2idx = tokenizer.word2idx # token to id map
idx2word = tokenizer.idx2word # id to token map
vocab_size = len(tokenizer.word2idx)
pad_token_id = tokenizer.word2idx["<PAD>"] # padding token

# dataset wrapper for article-summary pairs
class SummarizationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    # number of samples
    def __len__(self):
        return len(self.pairs)

    # get one sample
    def __getitem__(self, idx):
        return self.pairs[idx]


# pad sequences in a batch to same length
def collate_fn(batch):

    # split articles and summaries
    articles  = [item[0] for item in batch]
    summaries = [item[1] for item in batch]

    # longest sequence lengths in batch
    max_a = max(len(a) for a in articles)
    max_s = max(len(s) for s in summaries)

    # pad articles
    enc_art = [a + [pad_token_id] * (max_a - len(a)) for a in articles]

    # pad summaries
    enc_sum = [s + [pad_token_id] * (max_s - len(s)) for s in summaries]

    # return tensors
    return (
        torch.tensor(enc_art, dtype=torch.long),
        torch.tensor(enc_sum, dtype=torch.long),
    )

#set seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#hyperparameters
d_model = 512
num_layers_encoder= 6
num_layers_decoder = 6
max_len = 1024
MAX_ARTICLE_LEN = 512
MAX_SUMMARY_LEN = 128
num_heads = 8
dropout = 0.1
d_ff = 4 * d_model
batch_size = 32
number_of_epochs = 20
number_of_samples = 287000
label_smoothing = 0.1


# device using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

dataset = load_dataset("cnn_dailymail", "3.0.0") #dataset

#build encoder and decoder models
encoder = TransformerEncoder(
    vocab_size, d_model, num_heads, num_layers_encoder, d_ff, max_len, dropout
).to(device)

decoder = TransformerDecoder(
    d_model, num_layers_decoder, vocab_size, max_len, num_heads, dropout
)

# share decoder embedding and output weights
decoder.output_layer.weight = decoder.embedding.weight
decoder = decoder.to(device)
scaler = torch.amp.GradScaler(device=device, enabled=(device.type == "cuda"))

# optimizer with weight decay
def get_param_groups(encoder, decoder, weight_decay):
    #weight decay on weight matrices only we skip biases and LayerNorm
    decay, no_decay = [], []
    for model in [encoder, decoder]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

#using adamW optimizer
optimizer = torch.optim.AdamW(
    get_param_groups(encoder, decoder, weight_decay=1e-4),
    lr=2e-4,
    betas=(0.9, 0.98),
    eps=1e-9,
)

# cosine lr scheduler with warmup
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:  # linear warmup
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine decay after warmup
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# training and validation losses
loss_function = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=label_smoothing,
)
val_loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    reduction="mean",
)

# tokenize training set
print("Tokenising training data...")
pairs = tokenizer.get_training_pairs(dataset, limit=number_of_samples)
tokenized_pairs = []
for src, tgt in pairs:
    src_ids = tokenizer.encode_to_ids_article(
        src, tokenizer.merges, tokenizer.word2idx, max_len=MAX_ARTICLE_LEN)
    tgt_ids = tokenizer.encode_to_ids_summary(
        tgt, tokenizer.merges, tokenizer.word2idx, max_len=MAX_SUMMARY_LEN)
    tokenized_pairs.append((src_ids, tgt_ids))
print(f"Tokenised {len(tokenized_pairs)} training pairs")

# setup scheduler
total_steps = number_of_epochs * math.ceil(len(tokenized_pairs) / batch_size)
warmup_steps = total_steps // 20
scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
print(f"Scheduler: {total_steps} total steps, {warmup_steps} warmup steps")

# tokenize validation set
print("Tokenising validation data...")
val_pairs = tokenizer.get_validation_pairs(dataset, limit=5000)
tokenized_val_pairs = []
for src, tgt in val_pairs:
    src_ids = tokenizer.encode_to_ids_article(
        src, tokenizer.merges, tokenizer.word2idx, max_len=MAX_ARTICLE_LEN)
    tgt_ids = tokenizer.encode_to_ids_summary(
        tgt, tokenizer.merges, tokenizer.word2idx, max_len=MAX_SUMMARY_LEN)
    tokenized_val_pairs.append((src_ids, tgt_ids))
print(f"Tokenised {len(tokenized_val_pairs)} validation pairs")

# checkpoint paths
CHECKPOINT_PATH = "checkpoint_best.pth"
LATEST_PATH = "checkpoint_latest.pth"

# save model checkpoint
def save_checkpoint(epoch, val_loss, path):
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)
    print(f"  -> Checkpoint saved to {path} (epoch {epoch + 1}, val loss {val_loss:.4f})")

# resume training checkpoint
def load_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Resumed from {path} — epoch {ckpt['epoch'] + 1}, val loss {ckpt['val_loss']:.4f}")
    return ckpt["epoch"] + 1, ckpt["val_loss"]

# validation loop
def validate():
    encoder.eval()
    decoder.eval()
    total_loss, total_batches = 0.0, 0
    # no gradients during validation
    with torch.no_grad():
        # batch through validation set
        for start in range(0, len(tokenized_val_pairs), batch_size):
            batch_pairs = tokenized_val_pairs[start : start + batch_size]
            batch_articles = [p[0][:MAX_ARTICLE_LEN]  for p in batch_pairs]
            batch_summaries = [p[1][:MAX_SUMMARY_LEN]  for p in batch_pairs]

            max_a = max(len(a) for a in batch_articles)
            max_s = max(len(s) for s in batch_summaries)
            # pad validation batch
            enc_art = [x + [pad_token_id] * (max_a - len(x)) for x in batch_articles]
            enc_sum = [x + [pad_token_id] * (max_s - len(x)) for x in batch_summaries]

            article = torch.tensor(enc_art, dtype=torch.long).to(device)
            summary = torch.tensor(enc_sum, dtype=torch.long).to(device)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                encoder_pad_mask = (article != pad_token_id)
                decoder_input = summary[:, :-1]
                target = summary[:, 1:]
                decoder_pad_mask = (decoder_input != pad_token_id)

                encoder_output = encoder(article, pad_token_id=pad_token_id)
                logits = decoder(
                    decoder_input,
                    encoder_output,
                    padding_mask_for_self_attention=decoder_pad_mask,
                    padding_mask_for_cross_attention=encoder_pad_mask,
                )
                loss = val_loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0

# training loop
def train():
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5 # early stopping 
    if os.path.exists(LATEST_PATH):
        start_epoch, best_val_loss = load_checkpoint(LATEST_PATH)
    else:
        start_epoch = 0
    # rebuild scheduler after resuming
    global scheduler
    remaining_steps = (number_of_epochs - start_epoch) * math.ceil(len(tokenized_pairs) / batch_size)
    warmup_steps = remaining_steps // 20
    scheduler = get_lr_scheduler(optimizer, warmup_steps, remaining_steps)
    train_dataset = SummarizationDataset(tokenized_pairs)
    # create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    for epoch in range(start_epoch, number_of_epochs):
        # training mode
        encoder.train()
        decoder.train()

        epoch_loss = 0.0
        steps_this_epoch = 0

        for article, summary in train_loader:
            # move batch to gpu
            article = article.to(device, non_blocking=True)
            summary = summary.to(device, non_blocking=True)

            # mixed precision forward pass
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                # create padding masks
                encoder_pad_mask = (article != pad_token_id)
                # teacher forcing inputs and targets
                decoder_input = summary[:, :-1]
                target = summary[:, 1:]
                decoder_pad_mask = (decoder_input != pad_token_id)

                # forward pass
                encoder_output = encoder(article, pad_token_id=pad_token_id)
                logits = decoder(
                    decoder_input,
                    encoder_output,
                    padding_mask_for_self_attention=decoder_pad_mask,
                    padding_mask_for_cross_attention=encoder_pad_mask,
                )

                # compute loss
                loss = loss_function(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1)
                )

            # skip unstable batches
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            # backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 0.5
            )

            # optimizer step
            scaler.step(optimizer)
            scaler.update()
            # update learning rate
            scheduler.step()

            epoch_loss += loss.item()
            steps_this_epoch += 1

        avg_train_loss = epoch_loss / steps_this_epoch
        val_loss = validate() # run validation after each epoch
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{number_of_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # save latest checkpoint
        save_checkpoint(epoch, val_loss, path=LATEST_PATH)

        if val_loss < best_val_loss: # save best model
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(epoch, val_loss, path=CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                return best_val_loss

    print("Training complete.")
    return best_val_loss

if __name__ == "__main__":
    #start training
    best_val_loss = train()

    

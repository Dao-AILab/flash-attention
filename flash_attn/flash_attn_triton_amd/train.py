import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from flash_attn import flash_attn_qkvpacked_func, flash_attn_qkvpacked_fp8_func, flash_attn_varlen_qkvpacked_func, flash_attn_varlen_qkvpacked_fp8_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# -------------------------------
# Model
# -------------------------------
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, causal=True, dropout=0.1, qkv_bias=True, use_fp8=False):
        super().__init__()
        self.use_fp8 = use_fp8
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.dropout_p = dropout
        
        # qkv and output projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, c = x.shape
        # project to qkv
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # reshape for flash attention function
        qkv_packed = torch.stack([q, k, v], dim=2).reshape(b, n, 3, self.num_heads, self.head_dim)
        
        # use the appropriate flash attention function
        if self.use_fp8:
            context = flash_attn_qkvpacked_fp8_func(
                qkv_packed,
                dropout_p=self.dropout_p,
                causal=self.causal
            )
        else:
            context = flash_attn_qkvpacked_func(
                qkv_packed, 
                dropout_p=self.dropout_p,
                causal=self.causal
            )
        
        # convert back to original shape and type
        context = context.reshape(b, n, c)
        
        # output projection
        x = self.proj(context)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, causal=True, dropout=0.1, use_fp8=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads=num_heads, causal=causal, dropout=dropout, use_fp8=use_fp8)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FlashLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        causal=True,
        dropout=0.1,
        max_seq_len=256,
        use_fp8=False
    ):
        super().__init__()
        
        # embedding layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.dropout = nn.Dropout(dropout)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, causal=causal, dropout=dropout, use_fp8=use_fp8)
            for _ in range(depth)
        ])
        
        # lm head: project back to vocabulary dimension for each token
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        b, n = x.shape
        
        # token + positional embedding
        x = self.token_embedding(x)
        x = x + self.position_embedding[:, :n, :]
        x = self.dropout(x)
        
        # transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # language modeling head
        x = self.norm(x)
        logits = self.lm_head(x)  # shape: (b, n, vocab_size)
        return logits

# -------------------------------
# Data
# -------------------------------
class TextDataset(Dataset):
    def __init__(self, sequences, max_len=None):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # input: all tokens except the last, target: all tokens except the first
        return (torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:], dtype=torch.long))

class VarLenTextDataset(Dataset):
    def __init__(self, sequences, max_len=256):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Ensure the sequence doesn't exceed max_len+1
        seq = seq[:self.max_len+1]
        # input: all tokens except the last, target: all tokens except the first
        return (torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:], dtype=torch.long))

def prepare_dataset(batch_size, is_varlen=False, min_len=10, max_len=256, ratio_shorter=0.7):
    # load the WikiText-2
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # build vocabulary
    corpus = " ".join([line for line in dataset["text"] if line.strip() != ""]) # join non-empty lines into a single corpus string
    tokens = corpus.split()
    vocab = sorted(set(tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    token_ids = [word2idx[word] for word in tokens]
    
    num_workers = 2
    if is_varlen:
        # VARIABLE LENGTH: create sequences of different lengths
        sequences = []
        for i in range(0, len(token_ids) - max_len, max_len // 2):  # overlap to get more sequences
            # Decide target length for this sequence
            if np.random.random() < ratio_shorter:
                # Shorter sequence
                target_len = np.random.randint(min_len + 1, max_len + 1)
            else:
                # Full length sequence
                target_len = max_len + 1
                
            # Extract sequence up to target length or whatever's available
            seq_end = min(i + target_len, len(token_ids))
            seq = token_ids[i:seq_end]
            
            # Only keep sequences that are long enough
            if len(seq) > min_len + 1:  # +1 because we need both input and target
                sequences.append(seq)

        print(f"Created {len(sequences)} variable-length sequences")
        
        # Get some statistics
        lens = [len(seq) for seq in sequences]
        print(f"Sequence length stats: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")
        
        # split dataset
        num_samples = len(sequences)
        num_train = int(0.8 * num_samples)
        num_val = num_samples - num_train
        
        # Use appropriate dataset class based on whether we need variable length
        dataset_class = VarLenTextDataset
        train_sequences = sequences[:num_train]
        val_sequences = sequences[num_train:]
        
        train_dataset = dataset_class(train_sequences, max_len)
        val_dataset = dataset_class(val_sequences, max_len)


        # collate function
        def collate_fn(batch):
            """
            Collate function that creates a flat representation for variable length flash attention.
            """
            # Separate inputs and targets
            inputs, targets = zip(*batch)
            
            # Get sequence lengths
            seq_lens = torch.tensor([len(x) for x in inputs], dtype=torch.int32)
            
            # Concatenate inputs and targets into single tensors
            flat_inputs = torch.cat(inputs)
            flat_targets = torch.cat(targets)
            
            # Create cumulative sequence lengths tensor
            cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
            
            # Calculate max sequence length for this batch
            max_seqlen = seq_lens.max().item()
            
            return flat_inputs, flat_targets, seq_lens, cu_seqlens, max_seqlen

        # data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        # FIXED LENGTH: create sequences of length max_len+1
        sequences = []
        for i in range(0, len(token_ids) - max_len, max_len):
            seq = token_ids[i : i + max_len + 1]
            if len(seq) == max_len + 1:
                sequences.append(seq)

        # split dataset
        num_samples = len(sequences)
        num_train = int(0.8 * num_samples)
        num_val = num_samples - num_train
        train_dataset, val_dataset = random_split(TextDataset(sequences), [num_train, num_val])
    
        # data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    vocab_size = len(vocab)
    print(f"vocab size: {vocab_size}, train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    return train_dataloader, val_dataloader, vocab_size

# -------------------------------
# Training
# -------------------------------
def train_lm(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in tqdm(train_dataloader, desc=f"epoch {epoch+1}/{num_epochs} [train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_dataloader)
        train_losses.append(epoch_train_loss)
        print(f"epoch {epoch+1}/{num_epochs} - train loss: {epoch_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"epoch {epoch+1}/{num_epochs} [validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_dataloader)
        val_losses.append(epoch_val_loss)
        print(f"epoch {epoch+1}/{num_epochs} - validation loss: {epoch_val_loss:.4f}")
    
    return train_losses, val_losses

# -------------------------------
# Main
# -------------------------------
def main():
    # hyperparameters
    batch_size = 16
    num_epochs = 20
    learning_rate = 3e-4
    max_len = 128 # total length including both input and target tokens
    is_varlen = False
    causal=True
    dropout=0.1
    
    # prep data
    print("Preparing Dataset")
    train_dataloader, val_dataloader, vocab_size = prepare_dataset(batch_size, max_len=max_len, is_varlen=is_varlen)
    
    # create language models
    print("Creating Models")
    model_normal = FlashLM(
        vocab_size=vocab_size,
        dim=256,
        depth=3,
        num_heads=8,
        causal=causal,
        dropout=dropout,
        max_seq_len=max_len,
    ).to(device)

    model_fp8 = FlashLM(
        vocab_size=vocab_size,
        dim=256,
        depth=3,
        num_heads=8,
        causal=causal,
        dropout=dropout,
        max_seq_len=max_len,
        use_fp8=True
    ).to(device)
    
    # Train Normal model
    print("Starting training for Normal model...")
    optimizer_normal = optim.AdamW(model_normal.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    normal_train_losses, normal_val_losses = train_lm(
        model_normal, train_dataloader, val_dataloader, optimizer_normal, criterion, num_epochs
    )
    torch.save(model_normal.state_dict(), 'flash_lm_normal.pth')
    print("Normal model training complete and saved.")

    # Train FP8 model
    print("Starting training for FP8 model...")
    optimizer_fp8 = optim.AdamW(model_fp8.parameters(), lr=learning_rate)
    fp8_train_losses, fp8_val_losses = train_lm(
        model_fp8, train_dataloader, val_dataloader, optimizer_fp8, criterion, num_epochs
    )
    torch.save(model_fp8.state_dict(), 'flash_lm_fp8.pth')
    print("FP8 model training complete and saved.")

    # save losses to csv
    epochs = range(1, num_epochs+1)
    loss_data = {
        "Epoch": epochs,
        "Normal_Training_Loss": normal_train_losses,
        "Normal_Validation_Loss": normal_val_losses,
        "FP8_Training_Loss": fp8_train_losses,
        "FP8_Validation_Loss": fp8_val_losses,
    }
    df_losses = pd.DataFrame(loss_data)
    df_losses.to_csv("losses.csv", index=False)
    print("Loss data saved to losses.csv")
    
    # plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, normal_train_losses, label="Normal Training Loss", marker='o')
    plt.plot(epochs, fp8_train_losses, label="FP8 Training Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison: Normal vs FP8 Flash Attention")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")  # Saves the training loss plot to disk
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, normal_val_losses, label="Normal Validation Loss", marker='o')
    plt.plot(epochs, fp8_val_losses, label="FP8 Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison: Normal vs FP8 Flash Attention")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_loss.png")  # Saves the validation loss plot to disk
    plt.show()


if __name__ == "__main__":
    main()

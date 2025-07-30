import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import random
from transformers import GPT2Tokenizer

# ----------------------------------------------------------


class CausalSelfAttention(nn.Module):

	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0

		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # key, query, value projections for all heads but in a batch. Saves you from three separate instantiations of nn.Linear
		self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection
		self.c_proj.NANOGPT_SCALE_INIT = 1 # set flag so we know on initialization we need to scale down the std for these residual streams

		self.n_head = config.n_head
		self.n_embd = config.n_embd

		self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, attention_mask=None):

		B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)

		# Calculate query, key, value for all heads in batch, move head forward in the shape to be a batch dim alongside B
		# nh is "number of heads", hs is "head size", and C is number of channels (nh * hs)
		# e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs = 768 channels in the Transformer

		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		# attention materializes the large (T, T) matrix for all queries and keys
		att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # --> (B, nh, T, T)

		# apply causal mask
		att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

		# apply padding mask if needed
		if attention_mask is not None:
			attention_mask = attention_mask[:, None, None, :] # (B, T) --> (B, 1, 1, T)
			att = att.masked_fill(attention_mask == 0, float('-inf'))

		att = F.softmax(att, dim=-1)
		att = self.dropout(att)

		y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output project
		y = self.c_proj(y)
		return y


class MLP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # On naming (eg 'c_fc'), we are replicating the GPT2 model
		self.gelu = nn.GELU(approximate='tanh')
		self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
		self.c_proj.NANOGPT_SCALE_INIT = 1 # set flag so we know on initialization we need to scale down the std for these residual streams
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x


class Block(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x, attention_mask=None):
		x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
		x = x + self.mlp(self.ln_2(x))
		return x



@dataclass
class GPTConfig:
	block_size: int = 1024 # max sequence length
	vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
	n_layer: int = 12 # number of layers
	n_head: int = 12 # number of heads
	n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embd),
			wpe = nn.Embedding(config.block_size, config.n_embd),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = nn.LayerNorm(config.n_embd)
		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# weight sharing scheme
		self.transformer.wte.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5 # Scale down the residual streams so std doesn't bloat as the streams add. Note we multiply by 2 bc it happens twice in each Block (one residual in attention, one in MLP)
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def resize_token_embeddings_and_tie_weights(self, new_vocab_size):

		old_weight = self.transformer.wte.weight.data
		old_vocab_size, n_embd = old_weight.shape # Get current size

		assert new_vocab_size > old_vocab_size, f"New vocab size is not larger than current vocab size"

		# Create new embedding layer and copy weights
		self.transformer.wte = nn.Embedding(new_vocab_size, n_embd)
		self.transformer.wte.weight.data[:old_vocab_size] = old_weight
		# nn.init.normal_(self.transformer.wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)

		# Comment this out to illustrate bad initialization
		with torch.no_grad():
			average = self.transformer.wte.weight[:old_vocab_size].mean(dim=0) # Average of all embeddings across rows (vocab_size, n_embd) --> (n_embd)
			self.transformer.wte.weight.data[old_vocab_size:] = average

		# Create new lm_head layer
		self.lm_head = nn.Linear(n_embd, new_vocab_size, bias=False)

		# Tie weights
		self.lm_head.weight = self.transformer.wte.weight

		print(f"Model resized {model.transformer.wte} and {model.lm_head} layers to {new_vocab_size}")

	def forward(self, idx, targets=None, attention_mask=None):
		# idx is shape (B, T)
		B, T = idx.size()
		assert T <= self.config.block_size, f"Cannot forward sequence of length {T}. Block size is only {self.config.block_size}"

		# forward the token and position embeddings
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
		pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
		tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
		x = tok_emb + pos_emb

		# forward through the blocks of the transformer
		for block in self.transformer.h:
			x = block(x, attention_mask=attention_mask)

		# forward the final layernorm and the classifier
		x = self.transformer.ln_f(x)
		logits = self.lm_head(x) # (B, T, vocab_size)

		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		return logits, loss


	@classmethod
	def from_pretrained(cls, model_type):
		"""Loads pretrained GPT-2 model weights from huggingface"""
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		from transformers import GPT2LMHeadModel
		print("loading weights from pretrained gpt: %s" % model_type)

		# n_layer, n_head and n_embd are determined from model_type
		config_args = {
			'gpt2':			dict(n_layer=12, n_head=12, n_embd=768), 	# 124M params
			'gpt2-medium':	dict(n_layer=24, n_head=16, n_embd=1024), 	# 350M params
			'gpt2-large':	dict(n_layer=36, n_head=20, n_embd=1280), 	# 774M param
			'gpt2-xl':		dict(n_layer=48, n_head=25, n_embd=1600), 	# 1558M params
		}[model_type]
		config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
		config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints

		# create a from-scratch initialized minGPT model
		config = GPTConfig(**config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # dicard the mask / buffer

		# init a huggingface/transformers model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		# copy while ensuring all of the parameters are aligned and match in names and shapes
		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # dicard the mask / buffer
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # dicard the mask / buffer
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
		# basically the openai checkppoints use a "Conv1D" module, but we only want to use a vanilla Linear
		# this means we have to transpose these weights when we import them
		assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				# special treatment for the Conv1D weights we need to transpose
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())
			else:
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model

	def generate(self, device, max_length=50, num_return_sequences=1, query="Best tacos?", tokenizer=None):

		if tokenizer is None:
			from transformers import GPT2Tokenizer
			tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

		tokens = tokenizer.encode(query)
		tokens = torch.tensor(tokens, dtype=torch.long)
		tokens = tokens.unsqueeze(0)
		tokens = tokens.repeat(num_return_sequences,1)
		x = tokens
		x = x.to(device)

		while x.size(1) < max_length:
			with torch.no_grad():
				logits, loss = self(x)
				logits = logits[:, -1, :]
				probs = F.softmax(logits, dim=-1)
				topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
				ix = torch.multinomial(topk_probs, 1)
				xcol = torch.gather(topk_indices, -1, ix)
				x = torch.cat((x, xcol), dim=1)

				for i in range(num_return_sequences):
					tokens = x[i, :].tolist()
					decoded = tokenizer.decode(tokens)
					print(decoded)


class QADataLoader:
	def __init__(self, filepath, max_length=512, shuffle=True):
		self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		self.max_length = max_length
		self.shuffle = shuffle

		self.tokenizer.add_special_tokens({
			"bos_token": "<BOS>",
			"eos_token": "<EOS>",
			"sep_token": "<SEP>",
			"pad_token": "<PAD>"
		})

		self.special_tokens = {
			"<BOS>": self.tokenizer.encode("<BOS>")[0],
			"<SEP>": self.tokenizer.encode("<SEP>")[0],
			"<EOS>": self.tokenizer.encode("<EOS>")[0],
			"<PAD>": self.tokenizer.encode("<PAD>")[0]
		}

		self.samples = []
		with open(filepath, 'r') as f:
			for line in f:
				item = json.loads(line.strip())
				q, a = item["prompt"], item["completion"]
				tokens = self.encode_sample(q, a)
				if len(tokens["input_ids"]) <= self.max_length:
					self.samples.append(tokens)

		n = int(len(self.samples) * 0.9)

		self.train_data = self.samples[:n]
		self.val_data = self.samples[n:]

	def encode_sample(self, question, answer):
		q_tokens = self.tokenizer.encode(question)
		a_tokens = self.tokenizer.encode(answer)

		input_ids = (
			[self.special_tokens["<BOS>"]] +
			q_tokens +
			[self.special_tokens["<SEP>"]] +
			a_tokens +
			[self.special_tokens["<EOS>"]]
		)

		label_ids = input_ids[1:] + [-100] # Shift labels rightward by one to line up the labels
		ignore_length = len(q_tokens) + 1 # To account for q_token length and the <SEP> special token. Note the shift to the right above accounted for <BOS>
		label_ids[:ignore_length] = [-100] * ignore_length

		return {"input_ids": input_ids, "label_ids": label_ids}


	def __len__(self):
		return len(self.samples)

	def get_tokenizer(self):
		return self.tokenizer

	def get_batch(self, batch_size, split):

		data = self.train_data if split == 'train' else self.val_data

		if self.shuffle:
			batch = random.sample(data, batch_size)
		else:
			batch = data[:batch_size]

		max_len = max(len(sample["input_ids"]) for sample in batch)
		input_ids_batch = []
		label_ids_batch = []
		attention_mask_batch = []

		for sample in batch:
			pad_len = max_len - len(sample["input_ids"])
			input_ids = sample["input_ids"] + [self.special_tokens["<PAD>"]] * pad_len
			label_ids = sample["label_ids"] + [-100] * pad_len
			attention_mask = [1] * len(sample["input_ids"]) + [0] * pad_len

			input_ids_batch.append(input_ids)
			label_ids_batch.append(label_ids)
			attention_mask_batch.append(attention_mask)

		input_ids_batch = torch.tensor(input_ids_batch)
		label_ids_batch = torch.tensor(label_ids_batch)
		attention_mask_batch = torch.tensor(attention_mask_batch)

		return input_ids_batch, label_ids_batch, attention_mask_batch



# ----------------------------------------------------------

def save(model, optimizer, losses, filename):
	filename = f"{filename}.pt"
	directory = 'model/'
	os.makedirs(directory, exist_ok=True)
	filepath = os.path.join(directory, filename)
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': losses
	}
	torch.save(checkpoint, filepath)
	print(f"Model saved: {filepath}")


# ----------------------------------------------------------

torch.manual_seed(1337)
if torch.cuda.is_available():
	torch.cuda.manual_seed(1337)


dropout = 0.2
batch_size = 16
max_length = 512
model_save_path = '/model'


dataloader = QADataLoader("datasets/alpaca_data.jsonl", max_length=max_length)

model = GPT(GPTConfig())
model = model.from_pretrained('gpt2-medium')

tokenizer = dataloader.get_tokenizer()
new_vocab_length = len(tokenizer.get_vocab())

# resize embedding and final projection layers
model.resize_token_embeddings_and_tie_weights(new_vocab_length)

# set device and seed
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
	device = "mps"
print(f"using device: {device}")


torch.set_float32_matmul_precision('high')


# Send model to device
model.to(device)
use_compile = True
if use_compile:
	model = torch.compile(model)

epochs = 2
learning_rate = 1e-5
epsilon = 1e-8
max_steps = len(dataloader.samples) // batch_size

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon, weight_decay=0.05)

losses = {
	'training': [],
	'val': []
}

for epoch in range(epochs):
	for step in range(max_steps):
		t0 = time.time()

		# validation loss
		if step % 100 == 0:
			model.eval()
			with torch.no_grad():
				x, y, att_mask = dataloader.get_batch(batch_size, 'val')
				x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)
				logits, loss = model(x, y, att_mask)
				print(f"step {step}, validation loss: {loss.item()}")
				losses['val'].append(loss.item())

		# generate sample
		if step > 0 and step % 1000 == 0:
			model.eval()
			model.generate(max_length=30, device=device, tokenizer=tokenizer, query="List 3 properties of oxygen.")

		# train
		model.train()
		x, y, att_mask = dataloader.get_batch(batch_size, 'train')
		x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)
		optimizer.zero_grad()
		logits, loss = model(x, y, att_mask)
		loss.backward()
		optimizer.step()

		t1 = time.time()
		dt = t1 - t0
		tokens_processed = batch_size * x.size(-1) # Note that each batch size will have a dynamic length which is why we look at length of tokens in the batch which always gets the max length
		tokens_per_sec = tokens_processed / dt
		print(f"step {step} | training loss: {loss.item():.6f} | lr: {learning_rate:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
		losses['training'].append(loss.item())


save(model, optimizer, losses, "final")












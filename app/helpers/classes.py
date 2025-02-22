import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model configurations
with open("./app/models/bert-pretrained-data.pkl", "rb") as f:
    data = pickle.load(f)

n_layers = data["n_layers"]
n_heads = data["n_heads"]
d_model = data["d_model"]
d_ff = data["d_ff"]
d_k = data["d_k"]
d_v = data["d_v"]
n_segments = data["n_segments"]
vocab_size = data["vocab_size"]
word2id = data["word2id"]
batch_size = data["batch_size"]
max_mask = data["max_mask"]
max_len = data["max_len"]

# ---------------- Embedding Layer ---------------- #
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

# ---------------- Attention Mask ---------------- #
def get_attn_pad_mask(seq_q, seq_k):
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(seq_q.size(0), seq_q.size(1), seq_k.size(1))

# ---------------- Attention Layers ---------------- #
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_heads).to(device)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).to(device)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        return nn.LayerNorm(d_model).to(device)(nn.Linear(n_heads * d_v, d_model).to(device)(context) + Q), attn

# ---------------- BERT Model ---------------- #
class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding().to(device)
        self.layers = nn.ModuleList([MultiHeadAttention() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model).to(device)
        self.activ = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)

        # Decoder
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False).to(device)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab)).to(device)

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, _ = layer(output, output, output, attn_mask)

        h_pooled = self.activ(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled)  # Next Sentence Prediction

        # Masked Token Prediction
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(nn.Linear(d_model, d_model).to(device)(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return output, logits_lm, logits_nsp

# ---------------- Tokenizer ---------------- #
class SimpleTokenizer:
    def __init__(self, word2id):
        self.word2id = word2id
        self.word2id.setdefault("[UNK]", len(word2id))
        self.word2id.setdefault("[PAD]", len(word2id))
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.max_len = max_len

    def encode(self, sentences):
        output = {"input_ids": [], "attention_mask": []}
        unk_id = self.word2id["[UNK]"]
        pad_id = self.word2id["[PAD]"]

        for sentence in sentences:
            input_ids = [self.word2id.get(word, unk_id) for word in sentence.split()]
            input_ids = input_ids[:self.max_len] + [pad_id] * (self.max_len - len(input_ids))
            att_mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))

            output["input_ids"].append(torch.tensor(input_ids))
            output["attention_mask"].append(torch.tensor(att_mask))

        return output

    def decode(self, ids):
        return " ".join([self.id2word.get(int(idx), "[UNK]") for idx in ids])

# ---------------- Cosine Similarity ---------------- #
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# ---------------- Mean Pooling ---------------- #
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    return torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)

# ---------------- Similarity Calculation ---------------- #
def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    inputs_a = tokenizer.encode(sentence_a)
    inputs_b = tokenizer.encode(sentence_b)

    inputs_ids_a, attention_a = inputs_a["input_ids"][0].unsqueeze(0).to(device), inputs_a["attention_mask"][0].unsqueeze(0).to(device)
    inputs_ids_b, attention_b = inputs_b["input_ids"][0].unsqueeze(0).to(device), inputs_b["attention_mask"][0].unsqueeze(0).to(device)

    u, _, _ = model(inputs_ids_a, inputs_ids_a, inputs_ids_a)
    v, _, _ = model(inputs_ids_b, inputs_ids_b, inputs_ids_b)

    return cosine_similarity(mean_pool(u, attention_a).cpu().numpy(), mean_pool(v, attention_b).cpu().numpy())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pickle.load(open('./app/models/bert-pretrained-data.pkl', 'rb'))
n_layers = data['n_layers']
n_heads = data['n_heads']
d_model = data['d_model']
d_ff = data['d_ff']
d_k = data['d_k']
d_v = data['d_v']
n_segments = data['n_segments']
vocab_size = data['vocab_size']
word2id = data['word2id']
batch_size = data['batch_size']
max_mask = data['max_mask']
max_len = data['max_len']

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        # x, seg: (bs, len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (len,) -> (bs, len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
    
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn 
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_heads).to(device)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).to(device)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model).to(device)(context)
        return nn.LayerNorm(d_model).to(device)(output + residual), attn  # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding().to(device)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model).to(device)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)
        self.classifier = nn.Linear(d_model, 2).to(device)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False).to(device)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab)).to(device)

    def forward(self, input_ids, segment_ids, masked_pos):
        input_ids = input_ids.to(self.embedding.tok_embed.weight.device)
        segment_ids = segment_ids.to(self.embedding.tok_embed.weight.device)
        masked_pos = masked_pos.to(self.embedding.tok_embed.weight.device)

        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        
        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return output, logits_lm, logits_nsp


# class SimpleTokenizer:
#     def __init__(self, word2id):
#         self.word2id = word2id
#         self.id2word = {v: k for k, v in self.word2id.items()}
#       ncode(self, sentences):
#         output = {}
#         output['input_ids'] = []
#         output['attention_mask'] = []
#         for sentence in sentences:
#             input_ids = [self.word2id.get(word, self.word2id['[UNK]']) for word in sentence.split()]
#             n_pad = self.max_len - len(input_ids)
#             input_ids.extend([0] * n_pad)
#             att_mask = [1 if idx != 0 else 0 for idx in input_ids]  # Create attention mask
#             output['input_ids'].append(torch.tensor(input_ids))  # Convert to tensor
#             output['attention_mask'].append(torch.tensor(att_mask))  # Convert to tensor
#         return output

#     def decode(self, ids):
#         return ' '.join([self.id2word.get(idx.item(), '[UNK]') for idx in ids])

# class SimpleTokenizer:
#     def __init__(self, word2id, max_len=32):
#         # Ensure [UNK] token exists
#         if "[UNK]" not in word2id:
#             word2id["[UNK]"] = len(word2id)  # Assign next available index

#         self.word2id = word2id
#         self.id2word = {v: k for k, v in self.word2id.items()}
#         self.max_len = max_len  # Set max sequence length

#     def encode(self, sentences):
#         output = {'input_ids': [], 'attention_mask': []}
#         unk_id = self.word2id["[UNK]"]
#         pad_id = self.word2id["[PAD]"]
#         vocab_size = len(self.word2id)  # Get vocabulary size

#         for sentence in sentences:
#             input_ids = [self.word2id.get(word, unk_id) for word in sentence.split()]
#             input_ids = input_ids[:self.max_len]  # Truncate if too long
#             n_pad = self.max_len - len(input_ids)
#             input_ids.extend([pad_id] * n_pad)  # Pad sequence to max_len

#             att_mask = [1] * (self.max_len - n_pad) + [0] * n_pad  # 1s for words, 0s for padding

#             # 🔍 Debugging: Check for invalid indices
#             for idx in input_ids:
#                 if idx < 0 or idx >= vocab_size:
#                     raise ValueError(f"Invalid token index {idx} (Vocabulary size: {vocab_size})")

#             output['input_ids'].append(torch.tensor(input_ids))  # Convert to tensor
#             output['attention_mask'].append(torch.tensor(att_mask))  # Convert to tensor

#         return output

#     def decode(self, ids):
#         return ' '.join([self.id2word.get(int(idx), "[UNK]") for idx in ids])


class SimpleTokenizer:
    def __init__(self, word2id):
        self.word2id = word2id
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.max_len = max_len

    def encode(self, sentences):
        output = {}
        output['input_ids'] = []
        output['attention_mask'] = []
        for sentence in sentences:
            input_ids = [self.word2id.get(word, self.word2id['[UNK]']) for word in sentence.split()]
            n_pad = self.max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            att_mask = [1 if idx != 0 else 0 for idx in input_ids]  # Create attention mask
            output['input_ids'].append(torch.tensor(input_ids))  # Convert to tensor
            output['attention_mask'].append(torch.tensor(att_mask))  # Convert to tensor
        return output

    def decode(self, ids):
        return ' '.join([self.id2word.get(idx.item(), '[UNK]') for idx in ids])

# Create segment_ids tensor with shape (batch_size, max_len)
segment_ids = torch.tensor([0] * max_len).unsqueeze(0).repeat(batch_size, 1).to(device)

# Create masked_pos tensor with shape (batch_size, max_mask)
masked_pos = torch.tensor([0] * max_mask).unsqueeze(0).repeat(batch_size, 1).to(device)

def configurations(u,v):
    # build the |u-v| tensor
    uv = torch.sub(u, v)   # batch_size,hidden_dim
    uv_abs = torch.abs(uv) # batch_size,hidden_dim
    
    # concatenate u, v, |u-v|
    x = torch.cat([u, v, uv_abs], dim=-1) # batch_size, 3*hidden_dim
    return x

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    similarity = dot_product / (norm_u * norm_v)
    return similarity

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenizer.encode(sentence_a)
    inputs_b = tokenizer.encode(sentence_b)

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids'][0].unsqueeze(0).to(device)
    attention_a = inputs_a['attention_mask'][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b['input_ids'][0].unsqueeze(0).to(device)
    attention_b = inputs_b['attention_mask'][0].unsqueeze(0).to(device)

    # Extract token embeddings from BERT
    u, _, _ = model(inputs_ids_a, segment_ids, masked_pos)  # all token embeddings A = batch_size, seq_len, hidden_dim
    v, _, _ = model(inputs_ids_b, segment_ids, masked_pos)  # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score
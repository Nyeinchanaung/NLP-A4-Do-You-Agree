import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import seed, random, shuffle, randint
import pickle
import sys
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('./models/bert-pretrained-data.pkl', 'rb') as f:
    data = pickle.load(f)

n_layers = data['n_layers']
n_heads = data['n_heads']
d_model = data['d_model']
d_ff = data['d_ff']
d_k = data['d_k']
d_v = data['d_v']
n_segments = data['n_segments']
vocab_size = data['vocab_size']
word2id = data['word2id']
id2word = {v: k for k, v in word2id.items()}  # Added for tokenization
batch_size = data['batch_size']
max_mask = data['max_mask']
max_len = data['max_len']

# --- Existing Model Classes ---
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)     # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
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
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).to(device)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model).to(device)(context)
        return nn.LayerNorm(d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
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
        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return output, logits_lm, logits_nsp

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
            sentence = [x.lower() for x in sentence]
            sentence = [re.sub("[.,!?\\-]", '', x) for x in sentence]
            sentence = " ".join(sentence)
            input_ids = [self.word2id.get(word, self.word2id['[UNK]']) for word in sentence.split()]
            n_pad = self.max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            att_mask = [1 if idx != 0 else 0 for idx in input_ids]
            output['input_ids'].append(torch.tensor(input_ids))
            output['attention_mask'].append(torch.tensor(att_mask))
        return output

    def decode(self, ids):
        return ' '.join([self.id2word.get(idx.item(), '[UNK]') for idx in ids])

# --- Utility Functions ---
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    similarity = dot_product / (norm_u * norm_v)
    return similarity

# --- New Tokenization Function ---
def tokenize_sentence_model1(sentence_a, sentence_b):
    lst_input_ids_premise = []
    lst_input_ids_hypothesis = []
    lst_masked_tokens_premise = []
    lst_masked_pos_premise = []
    lst_masked_tokens_hypothesis = []
    lst_masked_pos_hypothesis = []
    lst_segment_ids = []
    lst_attention_premise = []
    lst_attention_hypothesis = []
    max_seq_length = 200
    seed(55)

    tokens_premise = [word2id[word] if word in word2id else word2id['[UNK]'] for word in sentence_a.split()]
    tokens_hypothesis = [word2id[word] if word in word2id else word2id['[UNK]'] for word in sentence_b.split()]
    
    input_ids_premise = [word2id['[CLS]']] + tokens_premise + [word2id['[SEP]']]
    input_ids_hypothesis = [word2id['[CLS]']] + tokens_hypothesis + [word2id['[SEP]']]
    
    segment_ids = [0] * max_seq_length
    
    n_pred_premise = min(max_mask, max(1, int(round(len(input_ids_premise) * 0.15))))
    candidates_masked_pos_premise = [i for i, token in enumerate(input_ids_premise) if token != word2id['[CLS]'] 
                                     and token != word2id['[SEP]']]
    shuffle(candidates_masked_pos_premise)
    masked_tokens_premise, masked_pos_premise = [], []
    for pos in candidates_masked_pos_premise[:n_pred_premise]:
        masked_pos_premise.append(pos)
        masked_tokens_premise.append(input_ids_premise[pos])
        if random() < 0.1:  # 10% replace with random token
            index = randint(0, vocab_size - 1)
            input_ids_premise[pos] = word2id[id2word[index]]
        elif random() < 0.8:  # 80% replace with [MASK]
            input_ids_premise[pos] = word2id['[MASK]']
        else:
            pass

    n_pred_hypothesis = min(max_mask, max(1, int(round(len(input_ids_hypothesis) * 0.15))))
    candidates_masked_pos_hypothesis = [i for i, token in enumerate(input_ids_hypothesis) if token != word2id['[CLS]'] 
                                        and token != word2id['[SEP]']]
    shuffle(candidates_masked_pos_hypothesis)
    masked_tokens_hypothesis, masked_pos_hypothesis = [], []
    for pos in candidates_masked_pos_hypothesis[:n_pred_hypothesis]:
        masked_pos_hypothesis.append(pos)
        masked_tokens_hypothesis.append(input_ids_hypothesis[pos])
        if random() < 0.1:  # 10% replace with random token
            index = randint(0, vocab_size - 1)
            input_ids_hypothesis[pos] = word2id[id2word[index]]
        elif random() < 0.8:  # 80% replace with [MASK]
            input_ids_hypothesis[pos] = word2id['[MASK]']
        else:
            pass

    # Pad input_ids to max_seq_length
    n_pad_premise = max_seq_length - len(input_ids_premise)
    input_ids_premise.extend([0] * n_pad_premise)
    # Attention mask for the full sequence (1 for tokens, 0 for padding)
    attention_premise = [1] * len(tokens_premise) + [0] * n_pad_premise
    if max_mask > n_pred_premise:
        n_pad_premise_mask = max_mask - n_pred_premise
        masked_tokens_premise.extend([0] * n_pad_premise_mask)
        masked_pos_premise.extend([0] * n_pad_premise_mask)
    # Ensure attention_premise is max_seq_length
    attention_premise.extend([0] * (max_seq_length - len(attention_premise)))

    n_pad_hypothesis = max_seq_length - len(input_ids_hypothesis)
    input_ids_hypothesis.extend([0] * n_pad_hypothesis)
    # Attention mask for the full sequence (1 for tokens, 0 for padding)
    attention_hypothesis = [1] * len(tokens_hypothesis) + [0] * n_pad_hypothesis
    if max_mask > n_pred_hypothesis:
        n_pad_hypothesis_mask = max_mask - n_pred_hypothesis
        masked_tokens_hypothesis.extend([0] * n_pad_hypothesis_mask)
        masked_pos_hypothesis.extend([0] * n_pad_hypothesis_mask)
    # Ensure attention_hypothesis is max_seq_length
    attention_hypothesis.extend([0] * (max_seq_length - len(attention_hypothesis)))

    lst_input_ids_premise.append(input_ids_premise)
    lst_input_ids_hypothesis.append(input_ids_hypothesis)
    lst_segment_ids.append(segment_ids)
    lst_masked_tokens_premise.append(masked_tokens_premise)
    lst_masked_pos_premise.append(masked_pos_premise)
    lst_masked_tokens_hypothesis.append(masked_tokens_hypothesis)
    lst_masked_pos_hypothesis.append(masked_pos_hypothesis)
    lst_attention_premise.append(attention_premise)
    lst_attention_hypothesis.append(attention_hypothesis)

    return {
        "premise_input_ids": lst_input_ids_premise,
        "premise_pos_mask": lst_masked_pos_premise,
        "hypothesis_input_ids": lst_input_ids_hypothesis,
        "hypothesis_pos_mask": lst_masked_pos_hypothesis,
        "segment_ids": lst_segment_ids,
        "attention_premise": lst_attention_premise,
        "attention_hypothesis": lst_attention_hypothesis,
    }

# --- New NLI Prediction Function ---
def predict_nli_and_similarity(model, sentence_a, sentence_b, device, classifier_head):
    model.eval()
    inputs = tokenize_sentence_model1(sentence_a, sentence_b)
    
    inputs_ids_a = torch.tensor(inputs['premise_input_ids']).to(device)
    pos_mask_a = torch.tensor(inputs['premise_pos_mask']).to(device)
    attention_a = torch.tensor(inputs['attention_premise']).to(device)
    inputs_ids_b = torch.tensor(inputs['hypothesis_input_ids']).to(device)
    pos_mask_b = torch.tensor(inputs['hypothesis_pos_mask']).to(device)
    attention_b = torch.tensor(inputs['attention_hypothesis']).to(device)
    segment = torch.tensor(inputs['segment_ids']).to(device)

    with torch.no_grad():
        u, _, _ = model(inputs_ids_a, segment, pos_mask_a)
        v, _, _ = model(inputs_ids_b, segment, pos_mask_b)

    u = mean_pool(u, attention_a)
    v = mean_pool(v, attention_b)

    u_np = u.cpu().numpy().reshape(-1)
    v_np = v.cpu().numpy().reshape(-1)

    #similarity_score = cosine_similarity(u_np.reshape(1, -1), v_np.reshape(1, -1))
    similarity_score = cosine_similarity(u_np, v_np)

    uv_abs = torch.abs(u - v)
    x = torch.cat([u, v, uv_abs], dim=-1)

    with torch.no_grad():
        logits = classifier_head(x)
        probabilities = F.softmax(logits, dim=-1)

    labels = ["Entailment", "Neutral", "Contradiction"]
    nli_result = labels[torch.argmax(probabilities).item()]

    return {"similarity_score": float(similarity_score), "nli_label": nli_result}

# --- Existing Functions Kept for Compatibility ---
def configurations(u, v):
    uv = torch.sub(u, v)
    uv_abs = torch.abs(uv)
    x = torch.cat([u, v, uv_abs], dim=-1)
    return x

def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    model.eval()
    inputs_a = tokenizer.encode([sentence_a])
    inputs_b = tokenizer.encode([sentence_b])
    inputs_ids_a = inputs_a['input_ids'][0].unsqueeze(0).to(device)
    attention_a = inputs_a['attention_mask'][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b['input_ids'][0].unsqueeze(0).to(device)
    attention_b = inputs_b['attention_mask'][0].unsqueeze(0).to(device)
    segment_ids = torch.tensor([0] * max_len).unsqueeze(0).to(device)
    masked_pos = torch.tensor([0] * max_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        u, _, _ = model(inputs_ids_a, segment_ids, masked_pos)
        v, _, _ = model(inputs_ids_b, segment_ids, masked_pos)
        
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)
    similarity_score = cosine_similarity(u.reshape(1, -1), v)
    return similarity_score

def tokenize_sentence_model(sentence_a, sentence_b):
    inputs = {
        "premise_input_ids": [101] + [word2id.get(word, word2id['[UNK]']) for word in sentence_a.split()] + [102],
        "hypothesis_input_ids": [101] + [word2id.get(word, word2id['[UNK]']) for word in sentence_b.split()] + [102],
        "premise_pos_mask": [1] * len(sentence_a.split()),
        "hypothesis_pos_mask": [1] * len(sentence_b.split()),
        "attention_premise": [1] * len(sentence_a.split()),
        "attention_hypothesis": [1] * len(sentence_b.split()),
        "segment_ids": [0] * len(sentence_a.split()),
    }
    return inputs
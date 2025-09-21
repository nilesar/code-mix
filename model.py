# # model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import XLMRobertaModel, XLMRobertaTokenizer

# class LingEncoder(nn.Module):
#     def __init__(self, vocab_size, wemb_dim=256, pos_size=50, pos_dim=32, ner_size=50, ner_dim=32, hidden=512, n_layers=2):
#         super().__init__()
#         self.word_embed = nn.Embedding(vocab_size, wemb_dim, padding_idx=1)
#         self.pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)
#         self.ner_embed = nn.Embedding(ner_size, ner_dim, padding_idx=0)
#         self.lstm = nn.LSTM(wemb_dim + pos_dim + ner_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)

#     def forward(self, src_ids, pos_ids, ner_ids):
#         w = self.word_embed(src_ids)  
#         p = self.pos_embed(pos_ids)
#         n = self.ner_embed(ner_ids)
#         x = torch.cat([w, p, n], dim=-1)
#         outputs, (hn, cn) = self.lstm(x)
#         return outputs  


# class GatedFeatureFusion(nn.Module):
#     def __init__(self, enc_dim, lm_dim):
#         super().__init__()
#         self.W_h = nn.Linear(enc_dim, enc_dim)
#         self.W_l = nn.Linear(lm_dim, enc_dim)
#         self.W_g = nn.Linear(enc_dim + lm_dim, enc_dim)

#     def forward(self, h, l):
#         # h: (B, T, enc_dim)
#         # l: (B, T, lm_dim)
#         h_star = torch.tanh(self.W_h(h))
#         l_star = torch.tanh(self.W_l(l))
#         g = torch.sigmoid(self.W_g(torch.cat([h, l], dim=-1)))
#         f = g * h_star + (1 - g) * l_star
#         return f


# class PointerGeneratorDecoder(nn.Module):
#     def __init__(self, vocab_size, emb_dim, enc_dim, dec_hidden):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
#         self.lstm = nn.LSTM(emb_dim + enc_dim, dec_hidden, batch_first=True)
#         self.W_gen = nn.Linear(dec_hidden + enc_dim + emb_dim, 1)
#         self.out = nn.Linear(dec_hidden, vocab_size)
#         self.enc_dim = enc_dim
#         self.dec_hidden = dec_hidden

#     def forward(self, tgt_ids, enc_outputs, src_ids, states):
#         # tgt_ids: (B, T_out)
#         # enc_outputs: (B, T_src, enc_dim)
#         # src_ids: (B, T_src) for copy
#         embedded = self.embed(tgt_ids)  # (B, T_out, emb_dim)
#         B, T_out, _ = embedded.size()
#         T_src = enc_outputs.size(1)
#         outputs = []
#         attn_weights_all = []
#         h, c = states  # each: (num_layers, B, H)
#         # for simplicity do stepwise decoding (can be vectorized)
#         prev = embedded[:,0,:].unsqueeze(1)  # assume teacher forcing with start token at pos0
#         for t in range(T_out):
#             # attention: score = dec_hidden · enc_outputs
#             # take last layer hidden
#             dec_h = h[-1].unsqueeze(1)  # (B,1,H)
#             # compute attention weights
#             attn_scores = torch.bmm(dec_h, enc_outputs.transpose(1,2)).squeeze(1)  # (B, T_src)
#             attn_weights = torch.softmax(attn_scores, dim=-1)
#             context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_dim)
#             # LSTM input: prev embedded + context
#             lstm_in = torch.cat([prev.squeeze(1), context], dim=-1).unsqueeze(1)
#             out, (h, c) = self.lstm(lstm_in, (h, c))
#             out = out.squeeze(1)  # (B, dec_hidden)
#             # generation distribution
#             vocab_dist = torch.softmax(self.out(out), dim=-1)  # (B, vocab)
#             # p_gen
#             p_gen_in = torch.cat([out, context, prev.squeeze(1)], dim=-1)
#             p_gen = torch.sigmoid(self.W_gen(p_gen_in))  # (B,1)
#             # copy distribution derived from attention weights mapped to vocabulary by summing attention mass for each source token id
#             # create copy_dist by zeroing then adding attention weights to corresponding token ids
#             copy_dist = torch.zeros_like(vocab_dist)
#             # scatter add
#             copy_dist = copy_dist.scatter_add(1, src_ids, attn_weights)  # src_ids must have same length T_src (B, T_src)
#             # final
#             final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist
#             outputs.append(final_dist.unsqueeze(1))  # (B,1,V)
#             attn_weights_all.append(attn_weights.unsqueeze(1))
#             # next input (teacher forcing): get embedded of ground truth or greedy from final_dist
#             if t+1 < T_out:
#                 prev = embedded[:, t+1, :].unsqueeze(1)
#         outputs = torch.cat(outputs, dim=1)  # (B, T_out, V)
#         return outputs, (h, c), torch.cat(attn_weights_all, dim=1)

# class CodeMixModel(nn.Module):
#     def __init__(self, tokenizer_vocab_size, lm_model_name='xlm-roberta-base', **kwargs):
#         super().__init__()
#         self.encoder = LingEncoder(vocab_size=tokenizer_vocab_size, **kwargs)
#         # XLM feature extractor
#         self.xlm = XLMRobertaModel.from_pretrained(lm_model_name)
#         self.gff = GatedFeatureFusion(enc_dim=2*kwargs.get('hidden',512), lm_dim=self.xlm.config.hidden_size)
#         self.decoder = PointerGeneratorDecoder(vocab_size=tokenizer_vocab_size, emb_dim=kwargs.get('wemb_dim',256),
#                                                enc_dim=2*kwargs.get('hidden',512), dec_hidden=kwargs.get('hidden',512))

#     def forward(self, src_ids, pos_ids, ner_ids, tgt_ids, attention_mask=None, xlm_tokenizer=None):
#         # encoder
#         enc_out = self.encoder(src_ids, pos_ids, ner_ids)  # (B, T_src, enc_dim)
#         # get lm features: use XLM tokenizer externally to produce XLM input ids / attention mask aligned to subword positions
#         # Here we assume src_ids correspond to tokenizer we used for XLM or you implement mapping pipeline externally.
#         # For simplicity, call xlm(...) with input_ids same as src_ids (ONLY if using same tokenizer); otherwise implement mapping.
#         xlm_feats = self.xlm(input_ids=src_ids, attention_mask=attention_mask).last_hidden_state
#         f = self.gff(enc_out, xlm_feats)
#         # initialize decoder states (zero)
#         B = src_ids.size(0)
#         h0 = torch.zeros(1, B, self.decoder.dec_hidden).to(src_ids.device)
#         c0 = torch.zeros(1, B, self.decoder.dec_hidden).to(src_ids.device)
#         outputs, states, attn = self.decoder(tgt_ids, f, src_ids, (h0, c0))
#         return outputs

# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel

class LingEncoder(nn.Module):
    def __init__(self, vocab_size, wemb_dim=256, pos_size=50, pos_dim=32, ner_size=50, ner_dim=32, hidden=512, n_layers=2):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, wemb_dim, padding_idx=1)
        self.pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)
        self.ner_embed = nn.Embedding(ner_size, ner_dim, padding_idx=0)
        # BiLSTM -> enc_dim = 2 * hidden
        self.lstm = nn.LSTM(wemb_dim + pos_dim + ner_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, src_ids, pos_ids, ner_ids):
        w = self.word_embed(src_ids)    # (B, T, wemb_dim)
        p = self.pos_embed(pos_ids)     # (B, T, pos_dim)
        n = self.ner_embed(ner_ids)     # (B, T, ner_dim)
        x = torch.cat([w, p, n], dim=-1)   # (B, T, wemb+pos+ner)
        outputs, (hn, cn) = self.lstm(x)   # outputs: (B, T, 2*hidden)
        return outputs


class GatedFeatureFusion(nn.Module):
    def __init__(self, enc_dim, lm_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, enc_dim)
        self.W_l = nn.Linear(lm_dim, enc_dim)
        self.W_g = nn.Linear(enc_dim + lm_dim, enc_dim)

    def forward(self, h, l):
        # h: (B, T, enc_dim), l: (B, T, lm_dim)
        h_star = torch.tanh(self.W_h(h))
        l_star = torch.tanh(self.W_l(l))
        g = torch.sigmoid(self.W_g(torch.cat([h, l], dim=-1)))
        f = g * h_star + (1 - g) * l_star
        return f


class PointerGeneratorDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, dec_hidden, pad_token_id=1):
        """
        enc_dim: encoder output dim (e.g., 2*hidden)
        dec_hidden: decoder hidden size
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        # We'll project encoder outputs to dec_hidden for attention
        self.enc_proj = nn.Linear(enc_dim, dec_hidden)
        # LSTM input will be: embedding + context (context has size dec_hidden)
        self.lstm = nn.LSTM(emb_dim + dec_hidden, dec_hidden, batch_first=True)
        self.W_gen = nn.Linear(dec_hidden + dec_hidden + emb_dim, 1)  # (dec_h, context, emb)
        self.out = nn.Linear(dec_hidden, vocab_size)
        self.enc_dim = enc_dim
        self.dec_hidden = dec_hidden
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def forward(self, tgt_ids, enc_outputs, src_ids, states, teacher_forcing=True):
        """
        tgt_ids: (B, T_out) token ids (for teacher forcing); if teacher_forcing False, pass previous predictions
        enc_outputs: (B, T_src, enc_dim)
        src_ids: (B, T_src) token ids in source (for copy)
        states: (h, c) each shaped (num_layers, B, dec_hidden)
        """
        device = enc_outputs.device
        embedded = self.embed(tgt_ids)  # (B, T_out, emb_dim)
        B, T_out, emb_dim = embedded.size()
        T_src = enc_outputs.size(1)

        # project encoder outputs: (B, T_src, dec_hidden)
        enc_proj = self.enc_proj(enc_outputs)

        outputs = []
        attn_weights_all = []

        h, c = states  # (num_layers, B, dec_hidden)
        # initial prev embedding: use embedding of <s> token (assume tgt_ids[:,0] is <s>)
        prev = embedded[:, 0, :].unsqueeze(1)  # (B,1,emb_dim)

        # prepare mask to zero out attention on pads in source (if any)
        # assume pad token id for source = self.pad_token_id
        src_pad_mask = (src_ids == self.pad_token_id)  # (B, T_src)
        # we will set large negative scores for pad positions

        for t in range(T_out):
            dec_h_last = h[-1].unsqueeze(1)  # (B,1,dec_hidden)
            # attention scores: (B, 1, dec_hidden) x (B, T_src, dec_hidden)^T -> (B,1,T_src)
            attn_scores = torch.bmm(dec_h_last, enc_proj.transpose(1, 2)).squeeze(1)  # (B, T_src)

            # mask pad positions
            if src_pad_mask is not None:
                attn_scores = attn_scores.masked_fill(src_pad_mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T_src)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_proj).squeeze(1)  # (B, dec_hidden)

            # LSTM input: concat(prev_embedding, context)
            lstm_in = torch.cat([prev.squeeze(1), context], dim=-1).unsqueeze(1)  # (B,1, emb_dim + dec_hidden)
            out, (h, c) = self.lstm(lstm_in, (h, c))  # out: (B,1,dec_hidden)
            out = out.squeeze(1)  # (B, dec_hidden)

            # generation distribution
            vocab_logits = self.out(out)  # (B, vocab)
            vocab_dist = torch.softmax(vocab_logits, dim=-1)  # (B, vocab)

            # p_gen
            p_gen_in = torch.cat([out, context, prev.squeeze(1)], dim=-1)  # (B, dec_hidden + dec_hidden + emb_dim)
            p_gen = torch.sigmoid(self.W_gen(p_gen_in))  # (B,1)

            # copy distribution: map attn_weights over source tokens to vocab positions
            # build zero tensor and scatter-add attention weights to corresponding token ids
            copy_dist = torch.zeros_like(vocab_dist)  # (B, vocab)

            # Ensure src_ids values are in valid range and long
            idx = src_ids.long()  # (B, T_src)
            # attn_weights: (B, T_src)
            # scatter_add expects src shaped like idx. Use scatter_add_(dim=1, index=idx, src=attn_weights)
            # but PyTorch requires src to have same shape as index; so do:
            copy_dist = copy_dist.scatter_add(1, idx, attn_weights)

            # final distribution
            final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist  # (B, vocab)

            outputs.append(final_dist.unsqueeze(1))  # (B,1,vocab)
            attn_weights_all.append(attn_weights.unsqueeze(1))  # (B,1,T_src)

            # next input: teacher forcing or greedy
            if teacher_forcing:
                if t + 1 < T_out:
                    prev = embedded[:, t + 1, :].unsqueeze(1)
            else:
                # greedy pick
                next_id = torch.argmax(final_dist, dim=-1)  # (B,)
                prev = self.embed(next_id).unsqueeze(1)  # (B,1,emb_dim)

        outputs = torch.cat(outputs, dim=1)  # (B, T_out, vocab)
        attn = torch.cat(attn_weights_all, dim=1)  # (B, T_out, T_src)
        return outputs, (h, c), attn


class CodeMixModel(nn.Module):
    def __init__(self, tokenizer_vocab_size, lm_model_name='xlm-roberta-base', pad_token_id=1, **kwargs):
        super().__init__()
        # kwargs should include hidden, wemb_dim, etc.
        hidden = kwargs.get('hidden', 512)
        wemb_dim = kwargs.get('wemb_dim', 256)
        # encoder produces enc_dim = 2 * hidden
        self.encoder = LingEncoder(vocab_size=tokenizer_vocab_size, wemb_dim=wemb_dim, hidden=hidden, **kwargs)
        # XLM feature extractor
        self.xlm = XLMRobertaModel.from_pretrained(lm_model_name)
        enc_dim = 2 * hidden
        lm_dim = self.xlm.config.hidden_size
        self.gff = GatedFeatureFusion(enc_dim=enc_dim, lm_dim=lm_dim)
        # decoder: enc_dim passed so it can be projected internally
        self.decoder = PointerGeneratorDecoder(vocab_size=tokenizer_vocab_size, emb_dim=wemb_dim, enc_dim=enc_dim, dec_hidden=hidden, pad_token_id=pad_token_id)

    def forward(self, src_ids, pos_ids, ner_ids, tgt_ids, attention_mask=None):
        """
        src_ids: (B, T_src)
        pos_ids, ner_ids: (B, T_src)
        tgt_ids: (B, T_tgt)
        attention_mask: optional mask for xlm call (B, T_src)
        """
        # encoder
        enc_out = self.encoder(src_ids, pos_ids, ner_ids)  # (B, T_src, enc_dim)

        # XLM features - NOTE: must ensure src_ids align with XLM tokenizer in your pipeline
        # If you can't guarantee that, compute xlm_feats externally and pass them here.
        xlm_feats = self.xlm(input_ids=src_ids, attention_mask=attention_mask).last_hidden_state  # (B, T_src, lm_dim)

        # gated fusion
        fused = self.gff(enc_out, xlm_feats)  # (B, T_src, enc_dim)

        # initialize decoder states (zeros)
        B = src_ids.size(0)
        h0 = torch.zeros(1, B, self.decoder.dec_hidden, device=src_ids.device)
        c0 = torch.zeros(1, B, self.decoder.dec_hidden, device=src_ids.device)

        outputs, states, attn = self.decoder(tgt_ids, fused, src_ids, (h0, c0), teacher_forcing=True)
        return outputs

import torch
import math
import argparse
import sys

from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder
import tokenizer as tokenizer_module

d_model            = 512
num_layers_encoder = 6
num_layers_decoder = 6
max_len            = 1024
MAX_ARTICLE_LEN    = 512
MAX_SUMMARY_LEN    = 128
num_heads          = 8
dropout            = 0.0
d_ff               = 4 * d_model
CHECKPOINT_PATH    = "checkpoint_best.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading tokenizer...")
tokenizer_module.load_tokenizer()
merges     = tokenizer_module.merges
word2idx   = tokenizer_module.word2idx
idx2word   = tokenizer_module.idx2word
vocab_size = len(word2idx)
print(f"Vocab size: {vocab_size}")

pad_id = word2idx["<PAD>"]
sos_id = word2idx["<SOS>"]
eos_id = word2idx["<EOS>"]
unk_id = word2idx["<UNK>"]


def load_models(checkpoint_path=CHECKPOINT_PATH):
    encoder = TransformerEncoder(
        vocab_size, d_model, num_heads, num_layers_encoder, d_ff, max_len, dropout
    ).to(device)

    decoder = TransformerDecoder(
        d_model, num_layers_decoder, vocab_size, max_len, num_heads, dropout
    )
    decoder.output_layer.weight = decoder.embedding.weight
    decoder = decoder.to(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"  Loaded — epoch {ckpt['epoch'] + 1}, val loss {ckpt['val_loss']:.4f}")

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def apply_repetition_penalty(logits, generated_ids, penalty=1.5):
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def get_banned_ngram_tokens(generated_ids, ngram_size=4):
    if len(generated_ids) < ngram_size:
        return set()
    banned = set()
    current_context = tuple(generated_ids[-(ngram_size - 1):])
    for i in range(len(generated_ids) - ngram_size + 1):
        past_context = tuple(generated_ids[i:i + ngram_size - 1])
        if past_context == current_context:
            banned.add(generated_ids[i + ngram_size - 1])
    return banned


def ban_special_tokens(logits):
    logits[pad_id] = float("-inf")
    logits[sos_id] = float("-inf")
    logits[unk_id] = float("-inf")
    return logits


def temperature_decode(encoder, decoder, article_ids,
                       repetition_penalty=1.5,
                       no_repeat_ngram_size=4,
                       min_summary_len=25,
                       max_summary_len=MAX_SUMMARY_LEN,
                       temperature=0.7):
    with torch.no_grad():
        src = torch.tensor([article_ids], dtype=torch.long).to(device)
        encoder_pad_mask = (src != pad_id)
        encoder_output   = encoder(src, pad_token_id=pad_id)

        generated = [sos_id]

        for _ in range(max_summary_len):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
            decoder_pad_mask = (tgt != pad_id)

            logits = decoder(
                tgt,
                encoder_output,
                padding_mask_for_self_attention=decoder_pad_mask,
                padding_mask_for_cross_attention=encoder_pad_mask,
            )

            next_token_logits = logits[0, -1, :].clone()

            # Ban n-grams
            banned = get_banned_ngram_tokens(generated, no_repeat_ngram_size)
            for token_id in banned:
                next_token_logits[token_id] = float("-inf")

            # Repetition penalty (on raw logits before softmax — correct)
            next_token_logits = apply_repetition_penalty(
                next_token_logits, generated, repetition_penalty
            )

            # Ban specials
            next_token_logits = ban_special_tokens(next_token_logits)

            # Enforce min length
            if len(generated) - 1 < min_summary_len:
                next_token_logits[eos_id] = float("-inf")

            # Temperature sampling
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break

            generated.append(next_token)

        tokens = tokenizer_module.ids_to_tokens(generated[1:], idx2word)
        return tokenizer_module.decode(tokens)


def beam_search_decode(encoder, decoder, article_ids,
                       beam_size=6,
                       repetition_penalty=1.5,
                       no_repeat_ngram_size=4,
                       min_summary_len=25,
                       max_summary_len=MAX_SUMMARY_LEN,
                       length_penalty=0.9):
    with torch.no_grad():
        src = torch.tensor([article_ids], dtype=torch.long).to(device)
        encoder_pad_mask = (src != pad_id)
        encoder_output   = encoder(src, pad_token_id=pad_id)

        encoder_output   = encoder_output.expand(beam_size, -1, -1)
        encoder_pad_mask = encoder_pad_mask.expand(beam_size, -1)

        beams     = [([sos_id], 0.0, False)]
        completed = []

        completed_ids = set()

        def add_to_completed(beam):
            if id(beam) not in completed_ids:
                completed.append(beam)
                completed_ids.add(id(beam))

        for step in range(max_summary_len):
            if all(b[2] for b in beams):
                break

            active_beams = [(i, b) for i, b in enumerate(beams) if not b[2]]
            active_seqs  = [b[0] for _, b in active_beams]
            active_count = len(active_seqs)

            max_seq_len = max(len(s) for s in active_seqs)
            padded = [s + [pad_id] * (max_seq_len - len(s)) for s in active_seqs]

            tgt = torch.tensor(padded, dtype=torch.long).to(device)
            decoder_pad_mask = (tgt != pad_id)

            enc_out  = encoder_output[:active_count]
            enc_mask = encoder_pad_mask[:active_count]

            logits = decoder(
                tgt,
                enc_out,
                padding_mask_for_self_attention=decoder_pad_mask,
                padding_mask_for_cross_attention=enc_mask,
            )

            raw_logits = logits[:, -1, :]

            candidates = []

            for b in beams:
                if b[2]:
                    add_to_completed(b)

            for local_i, (_, beam) in enumerate(active_beams):
                seq, score, _ = beam

                token_logits = raw_logits[local_i].clone()

                # Ban n-grams
                banned = get_banned_ngram_tokens(seq, no_repeat_ngram_size)
                for tid in banned:
                    token_logits[tid] = float("-inf")

                # Repetition penalty (on raw logits — correct)
                for tid in set(seq):
                    if token_logits[tid] > 0:
                        token_logits[tid] /= repetition_penalty
                    else:
                        token_logits[tid] *= repetition_penalty

                # Ban specials
                token_logits = ban_special_tokens(token_logits)

                # Enforce min length
                if len(seq) - 1 < min_summary_len:
                    token_logits[eos_id] = float("-inf")

                # log_softmax after all logit-level modifications
                token_log_probs = torch.log_softmax(token_logits, dim=-1)

                top_log_probs, top_ids = torch.topk(token_log_probs, beam_size)

                for log_p, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                    new_seq   = seq + [tid]
                    new_score = score + log_p
                    finished  = (tid == eos_id)
                    candidates.append((new_seq, new_score, finished))

            if not candidates:
                break

            def score_fn(cand):
                seq, s, _ = cand
                length = max(len(seq) - 1, 1)
                return s / (length ** length_penalty)

            candidates.sort(key=score_fn, reverse=True)
            beams = candidates[:beam_size]

        # Post-loop: collect any remaining beams 
        for b in beams:
            add_to_completed(b)

        if not completed:
            completed = beams

        def final_score(cand):
            seq, s, _ = cand
            length = max(len(seq) - 1, 1)
            return s / (length ** length_penalty)

        best_seq = max(completed, key=final_score)[0]
        tokens   = tokenizer_module.ids_to_tokens(best_seq[1:], idx2word)
        return tokenizer_module.decode(tokens)



def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n(hypothesis, reference, n):
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
    matches = 0
    for ng in hyp_ngrams:
        if ref_counts.get(ng, 0) > 0:
            matches += 1
            ref_counts[ng] -= 1
    precision = matches / len(hyp_ngrams) if hyp_ngrams else 0.0
    recall    = matches / len(ref_ngrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(x, y):
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis, reference):
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs  = lcs_length(hyp_tokens, ref_tokens)
    prec = lcs / len(hyp_tokens)
    rec  = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_rouge(hypothesis, reference):
    return {
        "rouge-1": rouge_n(hypothesis, reference, 1),
        "rouge-2": rouge_n(hypothesis, reference, 2),
        "rouge-l": rouge_l(hypothesis, reference),
    }


def summarise(text, encoder, decoder,
              mode="beam",
              beam_size=6,
              repetition_penalty=1.5,
              no_repeat_ngram_size=4,
              min_summary_len=25,
              temperature=0.7):
    article = tokenizer_module.preprocess(text)
    ids     = tokenizer_module.encode_to_ids_article(
        article, merges, word2idx, max_len=MAX_ARTICLE_LEN
    )

    if mode == "greedy":
        # mode="greedy" kept for CLI back-compatibility;
        # internally calls temperature_decode (multinomial sampling)
        return temperature_decode(
            encoder, decoder, ids,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_summary_len=min_summary_len,
            temperature=temperature,
        )
    else:
        return beam_search_decode(
            encoder, decoder, ids,
            beam_size=beam_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_summary_len=min_summary_len,
        )


def evaluate_rouge(encoder, decoder, num_samples=200,
                   mode="beam", beam_size=6,
                   repetition_penalty=1.5,
                   no_repeat_ngram_size=4,
                   min_summary_len=25):
    from datasets import load_dataset
    dataset  = load_dataset("cnn_dailymail", "3.0.0")
    val_data = dataset["validation"]
    total    = min(num_samples, len(val_data))

    r1_total, r2_total, rl_total = 0.0, 0.0, 0.0
    print(f"\nEvaluating ROUGE on {total} validation samples ({mode} decoding)...\n")

    for i in range(total):
        article    = val_data[i]["article"]
        reference  = tokenizer_module.preprocess(val_data[i]["highlights"])
        hypothesis = summarise(
            article, encoder, decoder,
            mode=mode, beam_size=beam_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_summary_len=min_summary_len,
        )
        if i < 5: 
            print("\n" + "="*80)
            print(f"Sample {i+1}")
            print("-"*80)
            print("ARTICLE (truncated):")
            print(article[:500])
            print("\nREFERENCE SUMMARY:")
            print(reference)
            print("\nGENERATED SUMMARY:")
            print(hypothesis)
            print("="*80 + "\n")

        scores = compute_rouge(hypothesis, reference)
        r1_total += scores["rouge-1"]
        r2_total += scores["rouge-2"]
        rl_total += scores["rouge-l"]

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] "
                  f"R1={r1_total/(i+1):.4f}  "
                  f"R2={r2_total/(i+1):.4f}  "
                  f"RL={rl_total/(i+1):.4f}")

    print(f"\n── Final ROUGE scores ({total} samples) ──")
    print(f"  ROUGE-1 : {r1_total/total:.4f}")
    print(f"  ROUGE-2 : {r2_total/total:.4f}")
    print(f"  ROUGE-L : {rl_total/total:.4f}")

def interactive_demo(encoder, decoder):
    print("\n── Interactive Summarizer ──")
    print("Paste an article and press Enter twice to summarize.")
    print("Type 'quit' to exit.\n")

    while True:
        print("Article:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                sys.exit(0)
            if line.lower() == "quit":
                sys.exit(0)
            if line == "":
                break
            lines.append(line)

        if not lines:
            continue

        article = " ".join(lines)

        print("\nGenerating (beam search)...")
        beam_out = summarise(article, encoder, decoder, mode="beam")
        print(f"\n[Beam]        {beam_out}")

        print("\nGenerating (temperature sampling)...")
        temp_out = summarise(article, encoder, decoder, mode="greedy")
        print(f"\n[Temperature] {temp_out}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarization inference")
    parser.add_argument(
        "--mode", choices=["greedy", "beam", "eval", "interactive"],
        default="interactive",
    )
    parser.add_argument("--beam_size",            type=int,   default=6)
    parser.add_argument("--repetition_penalty",   type=float, default=1.5)
    parser.add_argument("--no_repeat_ngram_size", type=int,   default=4)
    parser.add_argument("--min_summary_len",      type=int,   default=25)
    parser.add_argument("--temperature",          type=float, default=0.7)
    parser.add_argument("--eval_samples",         type=int,   default=200)
    parser.add_argument("--checkpoint",           type=str,   default=CHECKPOINT_PATH)
    parser.add_argument("--article",              type=str,   default=None)
    args = parser.parse_args()

    encoder, decoder = load_models(args.checkpoint)

    if args.mode in ("greedy", "beam"):
        if args.article:
            text = args.article
        else:
            print("Paste article (end with empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            text = " ".join(lines)

        summary = summarise(
            text, encoder, decoder,
            mode=args.mode,
            beam_size=args.beam_size,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            min_summary_len=args.min_summary_len,
            temperature=args.temperature,
        )
        print(f"\nSummary:\n{summary}")

    elif args.mode == "eval":
        evaluate_rouge(
            encoder, decoder,
            num_samples=args.eval_samples,
            mode="beam",
            beam_size=args.beam_size,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            min_summary_len=args.min_summary_len,
        )

    elif args.mode == "interactive":
        interactive_demo(encoder, decoder)
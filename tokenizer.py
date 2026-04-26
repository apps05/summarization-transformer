import pickle
import os
#from datasets import load_dataset
from collections import defaultdict, Counter

#dataset = load_dataset("cnn_dailymail", "3.0.0")


def preprocess(text):
    text = text.strip()
    text = " ".join(text.split()) #basic whitespace cleanup
    return text


def simple_tokenize(text):
    tokens = []
    current = ""

    for ch in text:
        if ch.isalnum():
            current += ch 
        else:
            if current:
                tokens.append(current)
                current = ""   #reset current token
            if ch.strip():
                tokens.append(ch) #treating punctuation as separate tokens

    if current:
        tokens.append(current)

    return tokens


def get_training_corpus(dataset, batch_size=1000, limit=50000):
    train_data = dataset["train"]
    total = min(len(train_data), limit)

    for i in range(0, total, batch_size):
        batch = train_data[i:i+batch_size]
        texts = []
        for j in range(len(batch["article"])): # for each example in the batch
            texts.append(preprocess(batch["article"][j]))
            texts.append(preprocess(batch["highlights"][j]))
        yield texts # each batch is a list of preprocessed article and summary texts


def get_training_pairs(dataset, limit=50000):
    train_data = dataset["train"]
    total = min(len(train_data), limit)

    pairs = []
    for i in range(total):
        article = preprocess(train_data[i]["article"]) 
        summary = preprocess(train_data[i]["highlights"])
        pairs.append((article, summary))

    return pairs # list of (article, summary) tuples for training


def get_validation_pairs(dataset, limit=2000):
    """Uses the actual validation split — not the training set."""
    val_data = dataset["validation"]
    total = min(len(val_data), limit)

    pairs = []
    for i in range(total):
        article = preprocess(val_data[i]["article"])
        summary = preprocess(val_data[i]["highlights"])
        pairs.append((article, summary))

    return pairs # list of (article, summary) tuples from the validation set


# BPE training 

def build_vocab(corpus):
    vocab = Counter()
    for text in corpus:
        tokens = simple_tokenize(text)
        for token in tokens:
            if token.isalnum():
                chars = list(token) + ["</w>"] # adding end of word marker to distinguish "cat" from "c a t"
            else:
                chars = [token] # punctuation treated as single token words
            vocab[" ".join(chars)] += 1 # vocab keys are space-separated chars, values are frequencies
    return vocab #This allows BPE merges to combine chars into subwords


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq # count how many times each adjacent pair of symbols appears across the vocab
    return pairs #We use this to find the most common pair to merge next in the BPE algorithm


def merge_vocab(pair, vocab):
    new_vocab = {}
    for word, freq in vocab.items():
        symbols = word.split()
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2 # skip the next symbol since it's merged with the current one
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word = " ".join(new_symbols) 
        new_vocab[new_word] = new_vocab.get(new_word, 0) + freq #example: if pair is ('c', 'a') and word is "c a t </w>", it becomes "ca t </w>" in the new vocab with the same frequency 
    return new_vocab                                            #and if we have another word "c a r </w>", it also becomes "ca r </w>" and we sum frequencies for "ca t </w>" and "ca r </w>" in the new vocab
                                                                #we sum the freq of "ca t </w>" and "ca r </w>" because they both contain the merged pair "ca", which is what we are tracking in the BPE algorithm to find the most common pairs to merge next


def learn_bpe(corpus, num_merges=1000):
    
    vocab = build_vocab(corpus)

    base_chars = set()
    for word in vocab:
        for symbol in word.split():
            base_chars.add(symbol)   
    base_chars.add("</w>")           

    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get) # most frequent pair of symbols to merge next
        vocab = merge_vocab(best, vocab)
        merges.append(best)

        if (i + 1) % 100 == 0: #progress
            print(f"  BPE merges done: {i + 1}/{num_merges}")

    return merges, sorted(base_chars) #sorted base chars for consistent ordering when creating word2idx mapping



def create_id_maps(merges, base_chars):
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] #special tokens get reserved IDs at the start of the vocab
    merge_tokens   = ["".join(m) for m in merges]
    all_tokens     = special_tokens + list(base_chars) + merge_tokens
    unique_tokens  = list(dict.fromkeys(all_tokens))   # remove duplicates while preserving order

    word2idx = {token: i for i, token in enumerate(unique_tokens)} #unique ids for each token, with special tokens at the start of the vocab
    idx2word = {i: token for i, token in enumerate(unique_tokens)} #reverse mapping for decoding, also with special tokens at the start of the vocab

    return word2idx, idx2word



def apply_bpe(text, merges, word2idx):
    merge_ranks = {merge: i for i, merge in enumerate(merges)}
    tokens = simple_tokenize(text)
    output = []

    for token in tokens:
        if token.isalnum(): #if it's a normal word we break it down into chars + </w> for BPE processing
            word_tokens = []
            has_unk = False #flag to track if any part of the token is unknown which would prevent BPE merging and just return <UNK>
            for ch in token:
                if ch in word2idx:
                    word_tokens.append(ch)
                else:
                    word_tokens.append("<UNK>")
                    has_unk = True
            word_tokens.append("</w>")
        else: #otherwise we treat it as a single token, ex :- punctuation, and check if its in voacb
            if token in word2idx:
                word_tokens = [token]
            else:
                word_tokens = ["<UNK>"]
            has_unk = "<UNK>" in word_tokens

        if not has_unk: #if there is no UNK in word_tokens we can apply BPE merges to combine chars into subwords based on the learned merges
            while True:
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                best_pair = None
                best_rank = float("inf")
                for pair in pairs:
                    if pair in merge_ranks and merge_ranks[pair] < best_rank:
                        best_pair = pair
                        best_rank = merge_ranks[pair]

                if best_pair is None:
                    break

                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == best_pair
                    ):
                        new_tokens.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

        output.extend(word_tokens)

    return output #if there is UNK in word_tokens it return an empty list => entire token is UNK


def encode_to_ids_article(text, merges, word2idx, max_len=256):
    
    tokens = apply_bpe(text, merges, word2idx)
    tokens = tokens[:max_len - 2]
    unk_id = word2idx["<UNK>"]
    return ( #add SOS and EOS tokens and convert to IDs
        [word2idx["<SOS>"]]
        + [word2idx.get(t, unk_id) for t in tokens] #using UNK ID for any token not in vocab
        + [word2idx["<EOS>"]]
    )

#same thing for summary but max_len is 64 instead of 256 since summaries are shorter than articles
def encode_to_ids_summary(text, merges, word2idx, max_len=64):
   
    tokens = apply_bpe(text, merges, word2idx)
    tokens = tokens[:max_len - 2]
    unk_id = word2idx["<UNK>"]
    return (
        [word2idx["<SOS>"]]
        + [word2idx.get(t, unk_id) for t in tokens]
        + [word2idx["<EOS>"]]
    )



encode_to_ids = encode_to_ids_article # default encoding function for general use


def ids_to_tokens(ids, idx2word): #convert list of IDs back to tokens using idx2word mapping
    return [idx2word.get(i, "<UNK>") for i in ids] #UNK handling


def decode(tokens):
    words   = []
    current = ""

    for t in tokens:
        if t in ("<SOS>", "<EOS>", "<PAD>", "<UNK>"): #skip special tokens in decoding

            if current: 
                words.append(current) #add the current token
                current = "" #reset to ""
            continue

        if "</w>" in t:
            current += t.replace("</w>", "")
            words.append(current)#add token
            current = "" #reset for next token
        else:
            current += t #add to the current token being built up from BPE merges

    if current:
        words.append(current) #last token if it exists

    return " ".join(words) #join tokens with spaces to reconstruct the original text


def pad_batch(batch_ids, pad_id): #padding
    max_len = max(len(seq) for seq in batch_ids)
    return [seq + [pad_id] * (max_len - len(seq)) for seq in batch_ids]



merges   = None
word2idx = None
idx2word = None

#load saved tokenizer if it already exists
def load_tokenizer(cache_path="trained_tokenizer.pkl"):
    
    global merges, word2idx, idx2word

    if os.path.exists(cache_path):
        print("Loading tokenizer from cache...")
        with open(cache_path, "rb") as f:
            _data = pickle.load(f)

        merges   = _data["merges"]
        word2idx = _data["word2idx"]
        idx2word = _data["idx2word"]
        print(f"Tokenizer loaded. Vocab size: {len(word2idx)}")

    else:
        #train tokenizer once if no saved file is found
        print("Training tokenizer from scratch...")
        corpus = []
        for batch in get_training_corpus(dataset, limit=50000):
            corpus.extend(batch)

        #learn BPE merges and build token-id mappings
        merges, base_chars = learn_bpe(corpus, num_merges=8000)
        word2idx, idx2word = create_id_maps(merges, base_chars)

        #save tokenizer for future runs
        with open(cache_path, "wb") as f:
            pickle.dump({
                "merges":   merges,
                "word2idx": word2idx,
                "idx2word": idx2word,
            }, f)
        print(f"Tokenizer trained and saved. Vocab size: {len(word2idx)}")



if __name__ == "__main__":
    load_tokenizer()
    print("tokenizer ready")
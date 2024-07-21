# adpated from: https://huggingface.co/learn/nlp-course/en/chapter6/5
from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained('gpt2')

f_stoi = lambda s: {p:i for i,p in enumerate(s)}
f_itos = lambda s: {i:p for i,p in enumerate(s)}
def encode(s: str, stoi: dict):
    return [stoi[c] for c in s]
def decode(e, itos: dict):
    return ''.join([itos[i] for i in e])

def get_word_freqs(corpus: list):
    word_freqs = defaultdict(int)
    for txt in corpus:
        wrds_w_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
        nw_wrds = [word for word, offset in wrds_w_offsets]
        for word in nw_wrds: word_freqs[word] += 1
    return word_freqs

def get_a_v(word_freqs: defaultdict):
    alphabet = []
    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet: alphabet.append(letter)
    return alphabet.sort(), ['<E>'] + alphabet.copy()

def compute_pair_freqs(splits: dict, word_freqs: defaultdict):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1: continue
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a: str, b: str, splits: dict):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1: continue
        i = 0
        while i < len(split)-1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a+b] + split[i+2:]
            else: i += 1
        splits[word] = split
    return splits

def train(splits: dict, vocab: list, word_freqs: defaultdict, g_vocab_size: int):
    merges = defaultdict(str)
    while len(vocab) < g_vocab_size:
        print(f'{len(vocab)}/{g_vocab_size}', end='\r', flush=True)
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ''
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
    return splits, vocab, merges

def tokenize(txt: str, merges: defaultdict):
    ptres = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
    pttxt = [word for word, offset in ptres]
    splits = [[l for l in word] for word in pttxt]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split)-1:
                if split[i] == pair[0] and split[i+1] == pair[1]:
                    split = split[:i] + [merge] + split[i+2:]
                else: i += 1
            splits[idx] = split
    return sum(splits, [])

def detokenize(tkns: list):
    return ''.join(t.replace('Ġ', ' ') for t in tkns)

if __name__ == '__main__':
    with open('data/truths.txt', 'r', encoding='utf-8') as f: corpus = f.read().split('\n')
    word_freqs = get_word_freqs(corpus)
    _, vocab = get_a_v(word_freqs)
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    splits, vocab, merges = train(splits, vocab, word_freqs, 1000)

    stoi = f_stoi(vocab)
    print(tokenize('hello world', merges))
    print(encode(tokenize('hello world', merges), stoi))
    print(detokenize(tokenize('hello world', merges)))
    
    #with open('data/sample_truths.txt', 'r', encoding='utf-8') as f:
    #    txt = f.read()
    #    print(txt)
    #    print()
    #    txt_t = tokenize(txt, merges)
    #    # adding space just for visualization
    #    txt_t = [' ' + v if v.startswith('Ġ') else v for v in txt_t] 
    #    txt_t = ''.join(txt_t)
    #    print(txt_t)
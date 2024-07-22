from tokens import Tokens

if __name__ == '__main__':
    # https://gist.github.com/lukasnxyz/d7b29398dd3b3d1c1dcb14f1f19e3744
    # truths.txt has ~4861 unique words
    with open ('data/truths.txt', 'r', encoding='utf-8') as f: 
        corpus = f.read().split('\n') # input needs to be a list of strings
        tks = Tokens(corpus, 1000) # vocab, splits, merges are automatically processed

    test_str = 'the earth is round and so is the sun, but a cube is not.'
    tokenized = tks.tokenize(test_str)
    encoded = tks.encode(tokenized)

    print(tokenized)
    print(encoded)
    print(tks.decode(encoded))
    print(tks.detokenize(tks.decode(encoded))) 
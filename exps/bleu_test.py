from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# f = open('./valid.y.txt')
# hypo = f.readlines()
# def tokenize(x): return x.strip().split()
# map(tokenize, hypo)
# f.close()
# print corpus_bleu([[hypo]], [hypo])

# hypo = [[1, 2, 5, 7], [1, 4, 5, 6]]
# # ref = [[hypo[0]], [hypo[1]]]
# ref = map(lambda a: [a], hypo)  # wrap up ground truth label like this
# print corpus_bleu(ref, hypo)

hypo = ["he", "is", "PAD", "PAD", "PAD"]
ref = ["he", "is", "not", "PAD", "PAD"]
# we should only use cutomized weights
# padding will change BLEU, bumped from 0.606530659713 to 0.795270728767
# limit to at most tri-gram, but lower than 3 characters generation will get 0
print sentence_bleu([ref], hypo, weights=(0.25, 0.25))
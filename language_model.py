from itertools import count
from math import log, exp
from collections import defaultdict
import argparse

from numpy import mean

import nltk
from nltk import FreqDist
from nltk.util import bigrams
from nltk.tokenize import TreebankWordTokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:

    def __init__(self, unk_cutoff, jm_lambda=0.6, dirichlet_alpha=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower()):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._dirichlet_alpha = dirichlet_alpha
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        
        # Add your code here!
        self.vocab = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int) # is storing the counts of every two words that are seen in the training data. The key is a tuple of the two words, and the value is the count of how many times that pair has been seen.
        self.word_counts = defaultdict(int)
        self.successor_counts = defaultdict(int)
        self.predecessor_counts = defaultdict(int)
        self.total_count = 0
        self.total_unique_bigrams = 0

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here! 
        self.vocab[word] += count # every encountered word gets a plus 1 added to their vocab count          


    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # Add your code here
        if word in (kSTART, kEND):
            return word
        if self.vocab.get(word, 0) < self._unk_cutoff:
            return "<UNK>"
        return word

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True
        self.vocab_size = len(
            {self.vocab_lookup(word) for word in self.vocab}
            | {"<UNK>", kSTART, kEND}
        )

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)


    def mle(self, context, word):
        """
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        pair_count = self.bigram_counts[(context, word)]
        context_count = self.context_counts[context]

        if pair_count == 0 or context_count == 0:
            return kNEG_INF

        return lg(pair_count / context_count)

    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        pair_count = self.bigram_counts[(context, word)]
        context_count = self.context_counts[context]
        
        vocab_size = self.vocab_size
        prob = (pair_count + 1.0) / (context_count + vocab_size)
        return lg(prob)

    def jelinek_mercer(self, context, word):
        """
        OPTIONAL
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        pair_count = self.bigram_counts.get((context,word),0)
        context_count = self.context_counts.get(context,0)

        word_count = self.word_counts.get(word, 0)
        total_count = self.total_count
        if total_count == 0:
                            return kNEG_INF
        bigram_prob = pair_count/context_count if context_count > 0 else 0.0
        wordbigram = self._jm_lambda * bigram_prob + (1 - self._jm_lambda) * word_count/total_count
        if wordbigram ==0:
            return kNEG_INF
        return lg(wordbigram)

    def kneser_ney(self, context, word):
        """
        OPTIONAL
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        pair_count = self.bigram_counts[(context, word)]
        context_count = self.context_counts[context]

        # These statistics are accumulated once during training.
        unique_contexts_count = self.successor_counts.get(context, 0)
        continuation_count = self.predecessor_counts.get(word, 0)
        total_unique_bigrams = self.total_unique_bigrams

        if total_unique_bigrams == 0:
            return kNEG_INF

        vocab_size = self.vocab_size

        # The continuation distribution also uses discounting.
        # Words unseen as continuations back off to a uniform distribution.
        continuation_denominator = (
            total_unique_bigrams + self._kn_concentration
        )

        continuation_redistribution = (
            self._kn_concentration
            + self._kn_discount * len(self.predecessor_counts)
        ) / continuation_denominator

        continuation = (
            max(continuation_count - self._kn_discount, 0)
            / continuation_denominator
        ) + (
            continuation_redistribution * (1.0 / vocab_size)
        )

        # Apply discounting and concentration to the bigram probability
        context_denominator = context_count + self._kn_concentration

        redistribution = (
            self._kn_concentration
            + self._kn_discount * unique_contexts_count
        ) / context_denominator

        prob = (
            max(pair_count - self._kn_discount, 0)
            / context_denominator
        ) + (
            redistribution * continuation
        )

        if prob == 0:
            return kNEG_INF

        return lg(prob)

    def dirichlet(self, context, word):
        """
        OPTIONAL
        Additive smoothing, assuming independent Dirichlets with fixed
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        pair_count = self.bigram_counts[(context, word)]
        context_count = self.context_counts[context]
                
        vocab_size = self.vocab_size
        prob = (pair_count + self._dirichlet_alpha) / (context_count + (self._dirichlet_alpha * vocab_size))
        return lg(prob)

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of
        # code that will hopefully get you started.
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            if self.bigram_counts.get((context, word), 0) == 0:
                self.successor_counts[context] += 1
                self.predecessor_counts[word] += 1
                self.total_unique_bigrams += 1
            self.context_counts[context] += 1
            self.bigram_counts[(context, word)] += 1
            self.word_counts[word] += 1
            self.total_count += 1

    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))]))

    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.

        for ii in xrange(samples):
            yield ""
        return

# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--dir_alpha", help="Dirichlet parameter " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=kUNK_CUTOFF, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False,
                           choices=['mle', 'laplace', 'dirichlet',
                                    'jelinek_mercer', 'kneser_ney'])
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(args.unk_cutoff, jm_lambda=args.jm_lambda,
                             dirichlet_alpha=args.dir_alpha,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)

    for ii in nltk.corpus.brown.sents():
        for jj in lm.tokenize(" ".join(ii)):
            lm.train_seen(lm._normalizer(jj))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    sentence_count = 0
    for ii in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['kneser_ney', 'mle', 'dirichlet', \
                           'jelinek_mercer', 'good_turing', 'laplace'], \
      "Invalid estimation method"

    print("Evaluating Treebank sentences with %s..." % args.method, flush=True)
    results = []
    method = getattr(lm, args.method)
    for sentence_number, tokens in enumerate(nltk.corpus.treebank.sents(), start=1):
        sentence = " ".join(tokens)
        try:
            score = lm.perplexity(sentence, method)
        except OverflowError:
            score = float("inf")
        results.append((score, sentence))
        if sentence_number % 100 == 0:
            print("Evaluated %i sentences." % sentence_number, flush=True)

    results.sort()
    print("\nThree lowest-perplexity Treebank sentences:")
    for score, sentence in results[:3]:
        print(score, sentence)

    print("\nThree highest-perplexity Treebank sentences:")
    for score, sentence in reversed(results[-3:]):
        print(score, sentence)

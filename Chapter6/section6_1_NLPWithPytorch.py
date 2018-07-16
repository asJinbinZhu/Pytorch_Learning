# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
Part - 1 Introduction to Torch's tensor library
'''

# All of deep learning is computations on tensors, which are generalizations of a matrix that can be indexed in
# more than 2 dimensions.
# Creating Tensors.
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
# print(V)
'''
tensor([ 1.,  2.,  3.])
'''

M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data)
# print(M)
'''
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]])
'''

T_data = [
          [[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]
         ]
T = torch.Tensor(T_data)
# print(T)
'''
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]]])
'''

# print(V[0])
# print(M[0])
# print(T[0])
'''
tensor(1.)
tensor([ 1.,  2.,  3.])
tensor([[ 1.,  2.],
        [ 3.,  4.]])
'''

x = torch.randn((2, 3, 4))
# print(x)
'''
tensor([[[-1.5256, -0.7502, -0.6540, -1.6095],
         [-0.1002, -0.6092, -0.9798, -1.6091],
         [ 0.4391,  1.1712,  1.7674, -0.0954]],

        [[ 0.1394, -1.5785, -0.3206, -0.2993],
         [-0.7984,  0.3357,  0.2753,  1.7163],
         [-0.0561,  0.9107, -1.3924,  2.6891]]])
'''

# Operations with Tensors
x = torch.Tensor([1, 2, 3])
y = torch.Tensor([4, 5, 6])
z = x + y
# print(z)
'''
tensor([ 5.,  7.,  9.])
'''

x_1 = torch.randn((2, 5))
y_1 = torch.randn((3, 5))
z_1 = torch.cat([x_1, y_1]) # By default, it concatenates along the first axis (concatenates rows)
# print(z_1)
'''
tensor([[ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255],
        [-0.4791,  1.3790,  2.5286,  0.4107, -0.9880],
        [-0.9081,  0.5423,  0.1103, -2.2590,  0.6067],
        [-0.1383,  0.8310, -0.2477, -0.8029,  0.2366],
        [ 0.2857,  0.6898, -0.6331,  0.8795, -0.6842]])
'''

x_2 = torch.randn((2, 3))
y_2 = torch.randn((2, 5))
z_2 = torch.cat([x_2, y_2], 1)  # second arg specifies which axis to concat along
# print(x_2)
# print(y_2)
# print(z_2)
'''
tensor([[ 0.4533,  0.2912, -0.8317],
        [-0.5525,  0.6355, -0.3968]])
tensor([[-0.6571, -1.6428,  0.9803, -0.0421, -0.8206],
        [ 0.3133, -1.1352,  0.3773, -0.2824, -2.5667]])
        
tensor([[ 0.4533,  0.2912, -0.8317, -0.6571, -1.6428,  0.9803, -0.0421,
         -0.8206],
        [-0.5525,  0.6355, -0.3968,  0.3133, -1.1352,  0.3773, -0.2824,
         -2.5667]])
'''

# Reshaping Tensors
x = torch.randn((2, 3, 4))
# print(x)
# print(x.view(2, 12))     # Reshape to 2 rows, 12 columns
# print(x.view(2, -1))     # Same as above.  If one of the dimensions is -1, its size can be inferred
'''
tensor([[[ 1.5496,  0.3476,  0.0930,  0.6147],
         [ 0.7124, -1.7765,  0.3539,  1.1996],
         [ 0.4730, -0.4286,  0.5514, -1.5474]],

        [[ 0.7575, -0.4068, -0.1277,  0.2804],
         [ 1.7460,  1.8550, -0.7064,  2.5571],
         [ 0.7705, -1.0739, -0.2015, -0.5603]]])
         
tensor([[ 1.5496,  0.3476,  0.0930,  0.6147,  0.7124, -1.7765,  0.3539,
          1.1996,  0.4730, -0.4286,  0.5514, -1.5474],
        [ 0.7575, -0.4068, -0.1277,  0.2804,  1.7460,  1.8550, -0.7064,
          2.5571,  0.7705, -1.0739, -0.2015, -0.5603]])
          
tensor([[ 1.5496,  0.3476,  0.0930,  0.6147,  0.7124, -1.7765,  0.3539,
          1.1996,  0.4730, -0.4286,  0.5514, -1.5474],
        [ 0.7575, -0.4068, -0.1277,  0.2804,  1.7460,  1.8550, -0.7064,
          2.5571,  0.7705, -1.0739, -0.2015, -0.5603]])
'''

'''
Part - 2 Computation Graphs and Automatic Differentiation
'''
x = autograd.Variable(torch.Tensor([1, 2, 3]), requires_grad=True)
# print(x.data)
'''
tensor([ 1.,  2.,  3.])
'''
y = autograd.Variable(torch.Tensor([4, 5, 6]), requires_grad=True)
z = x + y
# print(z.data)
'''
tensor([ 5.,  7.,  9.])
'''

# print(z.grad_fn)    # Variables know what created them.
'''
<AddBackward1 object at 0x7f39d5396ad0>
'''

s = z.sum()
# print(s)
# print(s.grad_fn)
'''
tensor(21.)
<SumBackward0 object at 0x7f5f95c27b10>
'''

s.backward()
# print(x.grad)
'''
tensor([ 1.,  1.,  1.])
'''

x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y   # These are Tensor types, and backprop would not be possible

var_x = autograd.Variable(x, requires_grad=True)
var_y = autograd.Variable(y, requires_grad=True)
var_z = var_x + var_y
# print(var_z.grad_fn)
'''
<AddBackward1 object at 0x7fb7b8028cd0>
'''

var_z_data = var_z.data
new_var_z = autograd.Variable(var_z_data)   # does new_var_z have information to backprop to x and y? NO!
# print(new_var_z.grad_fn)
'''
None
'''

'''
Part - 3 Deep Learning Building Blocks: Affine maps, non-linearities and objectives
'''
# Affine Maps. 线性映射，即为f(x) = Ax + b
# for a matrix A and vectors x b. The parameters to be learned here are A and b. Often, b is refered to
# as the bias term.
lln = nn.Linear(5, 3)
data = autograd.Variable(torch.randn(2, 5))
# print(data)
# print(lln(data))
'''
tensor([[-0.7213,  2.2474, -1.3716, -0.8834,  0.5224],
        [-0.5692, -1.3981, -0.6388, -0.7874, -2.4998]])
        
tensor([[ 0.1406, -0.5901,  0.2199],
        [ 0.8363,  1.3558, -0.2581]])
'''
# print(lln.weight)
# print(lln.bias)
'''Parameter containing:
tensor([[-0.1118, -0.0362, -0.3853, -0.0883, -0.2884],
        [ 0.4109, -0.3866, -0.3486, -0.0152, -0.2418],
        [ 0.1600, -0.1721, -0.2101,  0.0253,  0.3237]])
Parameter containing:
tensor([-0.3146,  0.2100,  0.2873])

'''

# Non-Linearities. 非线性，常用的函数有 tanh(x),σ(x),ReLU(x) 这些都是激励函数 在pytorch中大部分激励函数在torch.functional中
data = autograd.Variable(torch.randn(2, 2))
# print(data)
# print(F.relu((data)))
'''
tensor([[ 0.1974, -2.4616],
        [-0.2671,  1.5170]])
tensor([[ 0.1974,  0.0000],
        [ 0.0000,  1.5170]])
'''
# Softmax and Probabilities.
# The function Softmax(x) is also just a non-linearity, but it is special in that it usually is the last operation
# done in a network. This is because it takes in a vector of real numbers and returns a probability distribution.

data = autograd.Variable(torch.randn(5))
# print(data)
# print(F.softmax(data))
# print(F.softmax(data).sum())    # Sums to 1 because it is a distribution!
# print(F.log_softmax(data))
'''
tensor([-1.9809,  0.5254,  0.3045,  0.8922,  2.1862])
tensor([ 0.0095,  0.1164,  0.0933,  0.1680,  0.6128])
tensor(1.)
tensor([-4.6569, -2.1507, -2.3715, -1.7838, -0.4898])
'''
'''
UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
Answer: softmax的隐式维度选择已经被弃用。更改调用包含dim = X作为参数
'''

# Objective Functions
# The objective function is the function that your network is being trained to minimize (in which case it is often
# called a loss function or cost function).

'''
Part - 4 Optimization and Training
'''
#  Well, since our loss is an autograd.Variable, we can compute gradients with respect to all of the parameters used
# to compute it! Then we can perform standard gradient updates.

'''
Part - 5 Creating Network Components in Pytorch
'''
# Before we move on to our focus on NLP, lets do an annotated example of building a network in Pytorch using only
# affine maps and non-linearities. We will also see how to compute a loss function, using Pytorch's built in negative
# log likelihood, and update parameters by backpropagation.

# Example: Logistic Regression Bag-of-Words classifier
data = [ ("me gusta comer en la cafeteria".split(), "SPANISH"),
         ("Give it to me".split(), "ENGLISH"),
         ("No creo que sea una buena idea".split(), "SPANISH"),
         ("No it is not a good idea to get lost at sea".split(), "ENGLISH") ]

test_data = [ ("Yo creo que si".split(), "SPANISH"),
              ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
'''
{'en': 3, 'No': 9, 'buena': 14, 'it': 7, 'at': 22, 'sea': 12, 'cafeteria': 5, 'Yo': 23, 'la': 4, 'to': 8, 'creo': 10, 
'is': 16, 'a': 18, 'good': 19, 'get': 20, 'idea': 15, 'qu
'''

label_to_ix = { "SPANISH": 0, "ENGLISH": 1 }

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):     # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec))


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function of a module,
# which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the Pytorch devs, your module (in this case, BoWClassifier)
# will store knowledge of the nn.Linear's parameters
'''
for param in model.parameters():
    print param
'''
'''
Parameter containing:
tensor([[-0.0807,  0.0478, -0.1371,  0.1289,  0.1229, -0.1556, -0.1611,
         -0.0172,  0.0824, -0.0057, -0.0994,  0.0045, -0.1843, -0.1386,
         -0.1306,  0.1615,  0.1729, -0.0666,  0.0088,  0.0875,  0.0235,
         -0.0982,  0.1131,  0.1206, -0.0114, -0.0241],
        [ 0.1782,  0.1714, -0.1112,  0.1919,  0.0485, -0.1303,  0.1074,
         -0.1464,  0.1812, -0.1261,  0.0555,  0.0597,  0.0466,  0.1627,
         -0.0815, -0.0828, -0.1699, -0.0080, -0.0929,  0.0079, -0.0402,
          0.0651,  0.1697,  0.0579, -0.0632, -0.0962]])
Parameter containing:
tensor([-0.1710,  0.1650])
'''

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
# print log_probs
'''
tensor([[-1.1426, -0.3842]])
'''

# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    # print log_probs
# print next(model.parameters())[:, word_to_ix["creo"]]   # Print the matrix column corresponding to "creo"
'''
tensor([[-0.9321, -0.5004]])
tensor([[-0.8059, -0.5918]])
tensor(1.00000e-02 *
       [-9.9446,  5.5467])
'''

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in xrange(100):
    for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a Variable
        # as an integer.  For example, if the target is SPANISH, then we wrap the integer
        # 0.  The loss function then knows that the 0th element of the log probabilities is
        # the log probability corresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    # print log_probs
# print next(model.parameters())[:, word_to_ix["creo"]] # Index corresponding to Spanish goes up, English goes down!
'''
tensor([[-0.1347, -2.0714]])
tensor([[-2.5193, -0.0839]])
tensor([ 0.3860, -0.4300])
'''
'''
the log probability for Spanish is much higher in the first example, and the log probability for English is much higher 
in the second for the test data, as it should be.
'''

'''
Part - 6 Word Embeddings: Encoding Lexical Semantics
'''
'''
uppose we are building a language model. Suppose we have seen the sentences

The mathematician ran to the store.
The physicist ran to the store.
The mathematician solved the open problem.
in our training data. Now suppose we get a new sentence never before seen in our training data:

The physicist solved the open problem.
Our language model might do OK on this sentence, but wouldn't it be much better if we could use the following two facts:

We have seen mathematician and physicist in the same role in a sentence. Somehow they have a semantic relation.
We have seen mathematician in the same role in this new unseen sentence as we are now seeing physicist.
'''

'''
Getting Dense Word Embeddings
'''
'''
How could we actually encode semantic similarity in words?
In summary, word embeddings are a representation of the semantics of a word, efficiently encoding semantic information 
that might be relevant to the task at hand. You can embed other things too: part of speech tags, parse trees, anything! 
The idea of feature embeddings is central to the field.
'''

'''
Word Embeddings in Pytorch
'''
word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5) # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix['hello']])
hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)
'''
tensor([[ 0.0276,  0.5652, -0.0115,  0.6706, -0.4929]])
'''

'''
An Example: N-Gram Language Modeling
'''
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
trigrams = [ ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in xrange(len(test_sentence) - 2) ]
# print trigrams[:3]
'''
[(['When', 'forty'], 'winters'), (['forty', 'winters'], 'shall'), (['winters', 'shall'], 'besiege')]
'''
vocab = set(test_sentence)
word_to_ix = { word: i for i, word in enumerate(vocab) }


class NGramLanguageModelerNGramLan(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModelerNGramLan, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModelerNGramLan(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in xrange(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = map(lambda w: word_to_ix[w], context)
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients.  Before passing in a new instance,
        # you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
# print losses  # The loss decreased every iteration over the training data!
'''
[tensor([ 518.0948]), tensor([ 515.5695]), tensor([ 513.0623]), tensor([ 510.5708]), tensor([ 508.0934]), 
tensor([ 505.6304]), tensor([ 503.1816]), tensor([ 500.7456]), tensor
'''


'''
Exercise: Computing Word Embeddings: Continuous Bag-of-Words
'''
CONTEXT_SIZE = 2 # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process. Computational processes are abstract
beings that inhabit computers. As they evolve, processes manipulate other abstract
things called data. The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
word_to_ix = { word: i for i, word in enumerate(set(raw_text)) }
data = []
for i in xrange(2, len(raw_text) - 2):
    context = [ raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2] ]
    target = raw_text[i]
    data.append( (context, target) )
# print data[:5]
'''
[
    (['We', 'are', 'to', 'study'], 'about'), 
    (['are', 'about', 'study', 'the'], 'to'), 
    (['about', 'to', 'the', 'idea'], 'study'), 
    (['to', 'study', 'idea', 'of'], 'the'), 
    (['study', 'the', 'of', 'a'], 'idea')
]
'''


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass


# create your model and train.  here are some functions to help you make the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = map(lambda w: word_to_ix[w], context)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# print(make_context_vector(data[0][0], word_to_ix))  # example
'''
tensor([ 13,   7,  18,  42])
'''

'''
Part - 7 Sequence Models and Long-Short Term Memory Networks
'''
# LSTM's in Pytorch
lstm = nn.LSTM(3, 3) # Input dim is 3, output dim is 3
inputs = [autograd.Variable(torch.randn((1, 3))) for _ in xrange(5)] # make a sequence of length 5
# print inputs
'''
[tensor([[-1.9851,  0.1355,  0.8472]]), tensor([[-1.4787, -1.4606,  0.4473]]), tensor([[ 0.4179, -0.5534, -0.4191]]), 
tensor([[ 1.0297, -0.1487, -1.0712]]), tensor([[ 0.3915, 1.2610,  0.3281]])] 
'''
test = autograd.Variable(torch.Tensor([[-1.9851,  0.1355,  0.8472]]))
# print(test)
# print(test.view(1, 1, -1))
'''
tensor([[-1.9851,  0.1355,  0.8472]])
tensor([[[-1.9851,  0.1355,  0.8472]]])
'''

# initialize the hidden state.
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))
# print(hidden)
'''
(tensor([[[ 0.1172, -1.6122,  0.1087]]]), tensor([[[ 1.9627, -0.9191, -0.9546]]]))
'''

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout the sequence.
# the second is just the most recent hidden state (compare the last slice of "out" with "hidden" below,
# they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropogate, by passing it as an argument
# to the lstm at a later time
inputs = torch.cat(inputs).view(len(inputs), 1, -1)  # Add the extra 2nd dimension
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
# print out
# print hidden
'''
tensor([[[ 0.5602, -0.0676, -0.1210]],

        [[ 0.1881,  0.0258, -0.1150]],

        [[ 0.1532,  0.0420,  0.1052]],

        [[ 0.1497, -0.0085,  0.2813]],

        [[ 0.0765, -0.1725,  0.3452]]])
(tensor([[[ 0.0765, -0.1725,  0.3452]]]), tensor([[[ 0.1548, -0.4072,  0.7118]]]))
'''

'''
Example: An LSTM for Part-of-Speech Tagging
'''


def prepare_sequence(seq, to_ix):
    idxs = map(lambda w: to_ix[w], seq)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
'''
{'Everybody': 5, 'ate': 2, 'apple': 4, 'that': 7, 'read': 6, 'dog': 1, 'book': 8, 'the': 3, 'The': 0}
'''
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# print(tag_scores)
'''
tensor([[-0.9622, -0.9866, -1.4061],
        [-0.9126, -1.0664, -1.3693],
        [-0.9591, -1.0079, -1.3792],
        [-0.9152, -1.0587, -1.3757],
        [-0.8746, -1.0985, -1.3879]])
'''

for epoch in xrange(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM, detaching it from its
        # history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j for word i.
# The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
# print(training_data[0][0])
# print(tag_scores)
'''
['The', 'dog', 'ate', 'the', 'apple']

tensor([[-0.4359, -1.3590, -2.3393],
        [-4.4667, -0.0409, -3.5563],
        [-3.1189, -3.0409, -0.0965],
        [-0.0341, -3.7047, -4.7218],
        [-2.8422, -0.0941, -3.4583]])
'''

# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# In the example above, each word had an embedding, which served as the inputs to our sequence model. Let's augment the
# word embeddings with a representation derived from the characters of the word. We expect that this should help
# significantly, since character-level information like affixes have a large bearing on part-of-speech. For example,
# words with the affix -ly are almost always tagged as adverbs in English.

'''
Part - 8 Advanced: Dynamic Toolkits, Dynamic Programming, and the BiLSTM-CRF
'''
'''
Pytorch is a dynamic neural network kit. The opposite is the static tool kit, which includes Theano, Keras, TensorFlow, 
etc. The core difference is the following:
In a static toolkit, you define a computation graph once, compile it, and then stream instances to it.
In a dynamic toolkit, you define a computation graph for each instance. It is never compiled and is executed on-the-fly
'''

# Bi-LSTM Conditional Random Field Discussion
'''
For this section, we will see a full, complicated example of a Bi-LSTM Conditional Random Field for named-entity 
recognition. 
'''

# The Forward Algorithm in Log-Space and the Log-Sum-Exp Trick


# Example: Bidirectional LSTM Conditional Random Field for Named-Entity Recognition
# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim / 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer *to* the start tag,
        # and we never transfer *from* the stop tag (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        self.hidden = self.init_hidden()
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        self.hidden = self.init_hidden()
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
# print model(precheck_sent)


# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in xrange(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()


# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# We got it!
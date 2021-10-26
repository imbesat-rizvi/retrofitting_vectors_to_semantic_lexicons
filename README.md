# retrofitting_vectors_to_semantic_lexicons

A vectorized iterative implementation of the paper ["Retrofitting Word Vectors to Semantic Lexicons"](https://aclanthology.org/N15-1184.pdf) in which the vector space representations are further refined using relational information from the semantic lexicons. The refined vectors were shown to have generally better performance on downstream semantic tasks compared to the original word vectors. The refining procedure is, however, agnostic to how the input vectors were constructed, and hence, can be used with any form of vectors (not necessarrily word vectors) given the semantic lexicon. 

A samll subset of sample files are presented in the examples folder. The program can be run as:

```
py retrofit.py --input example/sample_vec.txt --neighbours example/framenet.txt --output example/retrofitted_emb.txt --n_iter 10
```
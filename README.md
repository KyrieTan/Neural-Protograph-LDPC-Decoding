# Learning to decode Protograph LDPC Codes

Code used in paper [Learning to decode Protograph LDPC Codes]

Python environment:
       python = 3.6
       tensorflow = 2.0.0(use GPU for training)
       numpy =1.19.1
       (the versions are not too strict)

The file 'GeneratorMatrix.py' can get generator matrixs for every lifting factor by change the value of parmeter 'Z'.

The project contains five files which can train eight different neural LDPC algorithms in our paper.

           file                                           algorithm                                                                 hyperparmeter
    Neural_MS.py                           neural NOMS(type1)                                                             default
    
Neural_SP.py                                    neural SP                                                                          default

Neural_simplified_MS.py           simplified neural NOMS(type2)                      is_weight=True, is_bias=True, others default

Neural_simplified_MS.py           simplifeid neural NMS(type3)                         is_weight=True, is_bias=False, others default

Neural_simplified_MS.py           simplified neural NOMS(type4)                      is_weight=False, is_bias=False, others default

Neural_MS_damping.py          neural NOMS with damping(type5)                      single_damping=False, others default

Neural_MS_damping.py      simplified neural NOMS with damping(type6)           single_damping=True, others default

Neural_MS_multiloss.py        neural NOMS but not iteration-by-iteration                                   default

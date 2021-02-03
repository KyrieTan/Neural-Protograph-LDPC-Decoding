import numpy as np

# get the generator matrix of LDPC code
BaseGraph = 1
Z = 5            # lifting factor
ldpc_i_ls = 2     # index of base graph
code_BG = np.loadtxt("./BaseGraph/BaseGraph{0}_Set{1}.txt".format(BaseGraph, ldpc_i_ls), int, delimiter='	') # the matrix form of base graph
code_N = code_BG.shape[1]
code_m = code_BG.shape[0]
code_k = code_N - code_m
PCM = np.zeros([code_m * Z, code_N * Z], dtype=int)    # parity check matrix
for i in range(code_m):
    for j in range(code_N):
        if code_BG[i, j] != -1:
            a = code_BG[i, j] % Z
            for k in range(Z):
                PCM[i * Z + k, j * Z + (k + a) % Z] = 1

def roll_left(vec, L, Z):
    vec1 = np.zeros([Z], dtype=int)
    vec1[0:Z - L] = vec[L:Z]
    if L != 0:
        vec1[Z - L:] = vec[0:L]
    return vec1

def roll_right(vec, L, Z):
    vec1 = np.zeros([Z], dtype=int)
    vec1[0:L] = vec[Z-L:Z]
    vec1[L:Z] = vec[0:Z - L]
    if L != 0:
        vec1[0:L] = vec[Z-L:Z]
    return vec1

# 5G LDPC encoder
def LDPC_encoder(infoWord, code_PCM, code_n, code_m, Z):
    code_k = code_N - code_m
    infoWord = np.reshape(infoWord, [code_k, Z])
    encodeWord = np.zeros((code_n, Z), dtype=int)
    shift = np.zeros((code_m, code_N, Z), dtype=int)
    encodeWord[:code_k, :] = infoWord[:, :]
    for i in range(0, code_m, 1):
        for j in range(0, code_k, 1):
            if(code_PCM[i, j] != -1):
                shift[i, j, :] = roll_left(infoWord[j, :], code_PCM[i, j] % Z, Z)
    check_vec = np.sum(shift, axis=1) % 2
    encodeWord[code_k, :] = np.sum(check_vec[0:4, :], axis=0) % 2
    if(BaseGraph == 1 and ldpc_i_ls == 2):
        encodeWord[code_k, :] = roll_right(encodeWord[code_k, :], 105, Z)
    if(BaseGraph == 2 and ldpc_i_ls != 3 and ldpc_i_ls != 7):
        encodeWord[code_k, :] = roll_right(encodeWord[code_k, :], 1, Z)
    for i in range(0, code_m, 1):
        if (code_PCM[i, code_k] != -1):
            shift[i, code_k, :] = roll_left(encodeWord[code_k, :], code_PCM[i, code_k] % Z, Z)
    check_vec = np.add(check_vec, shift[:, code_k, :]) % 2
    for j in range(1, 4, 1):
        encodeWord[code_k + j, :] = check_vec[j - 1, :]
        for i in range(0, code_m, 1):
            if (code_PCM[i, code_k + j] != -1):
                shift[i, code_k + j, :] = roll_left(encodeWord[code_k + j, :], code_PCM[i, code_k + j] % Z, Z)
        check_vec = np.add(check_vec, shift[:, code_k + j, :]) % 2
    encodeWord[code_k + 4:code_n, :] = check_vec[4:code_m, :]
    encodeWord = np.reshape(encodeWord, [code_n * Z])
    return encodeWord

LDPC_G = np.zeros([code_k * Z, code_N * Z], dtype=int)
for i in range(0, code_k * Z, 1):
    infoWord = np.zeros(code_k * Z, dtype=int)
    infoWord[i] = 1
    LDPC_G[i, :] = LDPC_encoder(infoWord, code_BG, code_N, code_m, Z)

np.savetxt('./BaseGraph_GM/LDPC_GM_BG{0}_{1}.txt'.format(BaseGraph, Z), LDPC_G, fmt='%s', delimiter=',')

import numpy as np
from algo_options import parse_args

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def rand_model(n, d, disAxz, Ascale, xscale, zscale, seed):
    '''
    Generate random model b = A * x + z
    Inputs:
        n:      number of rows of A, which is the number of samples
        d:      number of columns of A, which is the dimension of x
        disAxz: distribution used to generate A, x, z:
                'n' stands for normal distribution; 
                'i' stands for integer distribution; 
                's' stands for sparse integer distribution; 
                'z' stands for zero distribution.
        Ascale: scalar to scale A
        xscale: scalar to scale x
        zscale: scalar to scale z
        seed:   random generator seed
    Outputs:
        A:      an n by d random matrix
        b:      b = A * x + z
        x:      original signal vector
    '''
    np.random.seed(seed)
    args = parse_args()

    # Generate A
    if disAxz[0] == 'n' or disAxz[0] == 'N':
        # generate a random matrix with normal distribution
        A = Ascale * np.random.randn(n, d)

    # Generate x
    if disAxz[1] == 'n' or disAxz[1] == 'N':
        # generate a random vector with normal distribution
        x = xscale * np.random.randn(d, 1)
    elif disAxz[1] == 'i' or disAxz[1] == 'I':
        # generate a random vector with uniform integer distribution in [-k, k]
        x = xscale * np.random.randint(-args.k + 1, args.k, size=(d, 1))
    elif disAxz[1] == 's' or disAxz[1] == 'S':
        # generate a random vector with sparse uniform integer distribution in [-k, k]
        x = xscale * np.random.randint(-args.k + 1, args.k, size=(d, 1))
        perturbation = np.random.permutation(d)
        '''TODO: make it a parameter'''
        sparsity = 10
        x[perturbation[:d-sparsity]] = 0

    # Generate z
    if disAxz[2] == 'n' or disAxz[2] == 'N':
        # generate a random vector with normal distribution 
        z = zscale * np.random.randn(n, 1)
    elif disAxz[2] == 'z' or disAxz[2] == 'Z': 
        # generate a zero vector
        z = np.zeros((n, 1))

    # generate b
    if args.loss == 'logistic':
        b = (sigmoid(A @ x + z) > 0.0).astype(int)
    elif args.loss == 'l2':
        b = A @ x + z

    return A, b, x

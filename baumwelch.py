import numpy as np

class HMM:
    def __init__(self, n,k):
        self.n = n
        self.k = k
        self.A = np.zeros((n, n))
        self.B = np.zeros((n,k))
        self.pi = np.zeros(n)

    def forward(self,O):
        T = len(O)

        alpha = np.zeros((T, n))

        # Compute the initial forward probabilities
        alpha[0] = pi * B[:, O[0]]

        # Compute the forward probabilities for each time step
        for t in range(1, T):
            for j in range(n):
                alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, O[t]]

        return alpha

    def backward(self, O):
        T = len(O)

        beta = np.zeros((T, n))

        # Set the last column of beta to 1
        beta[-1] = 1

        # Compute the backward probabilities for each time step
        for t in reversed(range(T-1)):
            for i in range(n):
                beta[t, i] = np.sum(A[i,:] * B[:,O[t+1]] * beta[t+1,:])

        return beta
    def baumWelch(self, O):
        alpha = self.forward(O)
        beta = self.backward(O)


if __name__ == "__main__":
    n = 3
    k = 2
    A = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.5, 0.2],
                  [0.1, 0.3, 0.6]])
    B = np.array([[0.9, 0.1],
                  [0.2, 0.8],
                  [0.4, 0.6]])
    pi = np.array([0.6, 0.2, 0.2])

    hmm = HMM(n,k)
    hmm.A = A
    hmm.B = B
    hmm.pi = pi

    O = np.array([0,1,0,1,0])
    print("forward")
    print(hmm.forward(O))
    print("backward")
    print(hmm.backward(O))

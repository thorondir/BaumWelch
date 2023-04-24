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

    def gamma(self, alpha, beta):
        return (alpha*beta)/np.sum(alpha[-1])

    def epsilon(self, alpha, beta, O):
        T = len(O)
        epsilon = np.zeros((T-1, len(A), len(A)))

        for t in range(T-1):
            for i in range(len(A)):
                for j in range(len(A)):
                    epsilon[t, i, j] = alpha[t,i]*A[i,j]*beta[t+1,j]*B[j,O[t+1]]
            epsilon[t] /= np.sum(epsilon, axis=(1,2))[t]

        return epsilon

    def update(self, O):
        T = len(O)

        alpha = self.forward(O)
        beta = self.backward(O)
        gamma = self.gamma(alpha, beta)
        epsilon = self.epsilon(alpha, beta, O)

        self.pi = gamma[0]
        self.A = np.sum(epsilon[0:T-1], axis=0)/np.sum(gamma[0:T-1], axis=0)

        barb = np.zeros(B.shape)
        for j in range(k):
            filter = [O[l]==j for l in range(len(O))]
            barb[:,j] = np.sum(gamma[filter], axis=0)/np.sum(gamma, axis=0)

        self.B = barb



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
    alpha = hmm.forward(O)
    print(alpha)
    print("backward")
    beta = hmm.backward(O)
    print(beta)

    hmm.update(O)
    print(hmm.A)
    print(hmm.B)
    print(hmm.pi)

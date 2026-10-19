from numpy.linalg import inv, solve
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / e_x.sum()

# LinUCB algorithm implementation
class LinUCB():

    def __init__(self, narms, ndims, alpha, device="cpu"):
        # Set number of arms
        self.narms = narms
        # Number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.alpha = alpha
        self.device = device  # Specify device (CPU/GPU)
        # Instantiate A as a ndims×ndims matrix for each arm
        self.A = np.zeros((self.narms, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((narms, self.ndims, 1))
        self.lr = 0.05
        self.epoch = 0.0
        self.reg=1e-6
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.narms):
            self.A[arm] = np.eye(self.ndims)

        super().__init__()
        return

    def play(self, context):
        # gains per each arm
        p_t = np.zeros(self.narms)
        alpha_scaled = self.alpha / (self.epoch / 10 + 1)

        for i in range(self.narms):
            # solve once per arm, reuse for both theta and uncertainty
            A_reg = self.A[i] + self.reg * np.eye(self.ndims)
            self.theta = solve(A_reg, self.b[i])
            cntx = context
            p_t[i] = self.theta.T.dot(cntx) + alpha_scaled * np.sqrt(
                cntx.dot(solve(A_reg, cntx)))
        action = np.random.choice(np.where(p_t == max(p_t))[0])
        return action, p_t

    def update(self, arm, reward, context):
        self.A[arm] = (1 - self.lr) * self.A[arm] + self.lr * np.outer(context, context)
        self.b[arm] = np.add((1 - self.lr) * self.b[arm].T, self.lr * context * reward).reshape(self.ndims, 1)
        return

    def update_epoch(self):
        self.epoch += 1
        return

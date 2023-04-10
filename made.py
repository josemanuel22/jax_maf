from typing import List, Optional
import jax.numpy as jnp
from jax import  nn, random, jit

class Sequential:
    def __init__(self, modules) -> None:
        self._modules = modules
        
    @jit
    def model(self, x):
        for layer in self._modules:
            x = layer(x)
        return x
    
    def modules(self):
        return self._modules
    
class MaskedLinear:
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.mask = None

    def initialise_mask(self, mask: jnp.ndarray):
        """Internal method to initialise mask."""
        self.mask = mask

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply masked linear transformation."""
        output = jnp.dot(x, self.mask * self.weight.T) + self.bias
        return output
    
class MADE:
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = 1234,
    ) -> None:
        """Initalise MADE model.

        Args:
            n_in: Size of input.
            hidden_dims: List with sizes of the hidden layers.
            gaussian: Whether to use Gaussian MADE. Default: False.
            random_order: Whether to use random order. Default: False.
            seed: Random seed for numpy. Default: None.
        """
        # Set random seed.
        self.key = random.PRNGKey(1234)
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        # List of layers sizes.
        dim_list = [self.n_in, *hidden_dims, self.n_out]
        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]),)
            self.layers.append(nn.relu)
        # Hidden layer to output layer.
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        # Create model.
        self.model = Sequential(self.layers)
        # Get masks for the masked activations.
        self._create_masks()

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        if self.gaussian:
            # If the output is Gaussian, return raw mus and sigmas.
            return self.model(x)
        else:
            # If the output is Bernoulli, run it through sigmoid to squash p into (0,1).
            return jax.nn.sigmoid(self.model(x))
        
            
    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        # Define some constants for brevity.
        L = len(self.hidden_dims)
        D = self.n_in

        # Whether to use random or natural ordering of the inputs.
        self.masks[0] = random.permutation(self.key, D) if self.random_order else jnp.arange(D)

        # Set the connectivity number m for the hidden layers.
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = random.randint(self.key, minval=low, maxval=D - 1, shape=(size,))

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = jnp.zeros((len(m_next), len(m)))
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M = M.at[(j, slice(None))].set(jnp.where(m_next[j] >= m, 1, 0))
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(jnp.concatenate((m, m), axis=0))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))

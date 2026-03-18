import jax
import jax.numpy as jnp
from flax import linen as nn

class FlaxAlexNet(nn.Module):
    """
    Flax implementation of AlexNet (Modernized PyTorch Version).
    """
    input_size: int = 224
    cin: int = 3
    cout: int = 1000
    dropout: float = 0.5
    features_only: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        """
        x: [bs, h, w, c] (JAX defaults to channel-last)
        train: Boolean for Dropout behavior
        """
        
        # 1. Architectural Style Logic
        if self.input_size > 64:
            # ImageNet options
            k1, s1 = (11, 11), (4, 4)
            k2 = (5, 5)
            pool_k, pool_s = (3, 3), (2, 2)
            p1, p2 = 'SAME', 'SAME' # Flax 'SAME' handles padding automatically
        else:
            # CIFAR-10 options
            k1, s1 = (3, 3), (1, 1)
            k2 = (3, 3)
            pool_k, pool_s = (2, 2), (2, 2)
            p1, p2 = 'SAME', 'SAME'

        # --- Features ---
        # Conv 1
        x = nn.Conv(features=64, kernel_size=k1, strides=s1, padding=p1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=pool_k, strides=pool_s)

        # Conv 2
        x = nn.Conv(features=192, kernel_size=k2, padding=p2)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=pool_k, strides=pool_s)

        # Conv 3, 4, 5
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=pool_k, strides=pool_s)

        if not self.features_only:
            # --- Adaptive Pooling ---
            # JAX doesn't have a direct 'AdaptiveAvgPool2d'. 
            # For AlexNet, we manually resize to 6x6.
            x = jax.image.resize(x, (x.shape[0], 6, 6, x.shape[-1]), method='bilinear')

            # --- Classifier ---
            x = x.reshape((x.shape[0], -1)) # Flatten: [bs, 256*6*6]

            # FC 1
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
            x = nn.Dense(features=4096)(x)
            x = nn.relu(x)

            # FC 2
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
            x = nn.Dense(features=4096)(x)
            x = nn.relu(x)

            # FC 3 (Output)
            x = nn.Dense(features=self.cout)(x)

        return x

if __name__ == "__main__":
    # Test with CIFAR-10 size
    model = FlaxAlexNet(input_size=32, cin=1, cout=10)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 32, 32, 1)) # [batch, height, width, channels]
    
    # Initialize parameters
    params = model.init(key, x)['params']
    y = model.apply({'params': params}, x)
    print(f"Output shape: {y.shape}") # (2, 10)

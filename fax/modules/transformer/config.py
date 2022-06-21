import dataclasses


@dataclasses.dataclass
class TransformerConfig:

  d_model: int
  d_k: int
  d_ff: int
  n_heads: int
  vocab_size: int
  n_layers: int
  max_len: int = 4096
  lambda_pe: float = 1.0
  lambda_e: float = dataclasses.field(init=False)
  tau: float = dataclasses.field(init=False)

  def __post_init__(self):
    self.lambda_e = self.d_model ** -0.5
    self.tau = 1 / self.d_k ** 0.5
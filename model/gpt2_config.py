# coding=utf-8
class GPT2Config():
    def __init__(
        self,
        vocab_size=7642,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        output_hidden_states=False,
        output_attentions=False,
        lr=3e-5,
        epoch=100,
        batch_size=2,
        dynamics_lr=False,
        read_len=40000,
        history_len=5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dynamics_lr = dynamics_lr
        self.read_len = read_len
        self.history_len = history_len

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

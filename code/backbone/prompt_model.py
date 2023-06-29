import torch


class ContinuousPrompt(torch.nn.Module):
    """
    refer to https://github.com/THUDM/P-tuning/blob/f4225d4d8003868b31ed659dfe9732a351ad42d9/PT-Fewshot/pet/wrapper.py#L564
    """
    def __init__(self, config_dict, init_embedding=None):
        super(ContinuousPrompt, self).__init__()

        self.prompt_length = config_dict["prompt_len"]
        self.embed_size = config_dict["embed_size"]
        self.hidden_size = self.embed_size
        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)
        self.init_embedding = init_embedding
        self.init()

    def init(self):
        if self.init_embedding is not None:
            with torch.no_grad():
                self.prompt_embeddings.weight.data = self.init_embedding.detach().clone()

    def forward(self):
        pass

class ContinuousPromptPool(torch.nn.Module):
    """
    refer to https://github.com/THUDM/P-tuning/blob/f4225d4d8003868b31ed659dfe9732a351ad42d9/PT-Fewshot/pet/wrapper.py#L564
    """
    def __init__(self, config_dict, init_emb):
        super(ContinuousPromptPool, self).__init__()

        self.K = config_dict["K"]
        self.prompt_length = config_dict["prompt_len"]
        self.embed_size = config_dict["embed_size"]
        with torch.no_grad():
            self.prompt_embeddings = torch.nn.Parameter(init_emb.contiguous().view(self.K, self.prompt_length * self.embed_size))
            self.keys = torch.nn.Parameter(torch.randn(self.K, self.embed_size))

    def forward(self):
        pass
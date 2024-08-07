import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


class ProntoMAV(torch.nn.Module):

    def __init__(self,
                 template="[1] [MASK] [2].",
                 special_tokens=None, #  ("[R1]", "[R2]")
                 pretrained_model="FacebookAI/roberta-large",
                 inner_dim=256,
                 *args, **kwargs):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model).to(device)
        self.template = template.replace("[MASK]", self.tokenizer.mask_token)

        self.pretrained_model = pretrained_model
        if "roberta" in self.pretrained_model.lower():
            self.plm_num_embeddings = self.model.roberta.embeddings.word_embeddings.num_embeddings
            self.plm_embedding_dim = self.model.roberta.embeddings.word_embeddings.embedding_dim
        else:
            self.plm_num_embeddings = self.model.bert.embeddings.word_embeddings.num_embeddings
            self.plm_embedding_dim = self.model.bert.embeddings.word_embeddings.embedding_dim

        for param in self.model.parameters():
            param.requires_grad = False

        self.inner_dim = inner_dim
        self.vocab_extractor = torch.nn.Linear(self.plm_num_embeddings, self.inner_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.layernorm = torch.nn.LayerNorm(self.plm_num_embeddings)
        self.out_proj = torch.nn.Linear(self.inner_dim, 1)

        self.special_tokens = None

        # SPECIAL TOKENS
        if special_tokens:
            self.special_tokens = special_tokens
            self.tokenizer.add_tokens(self.special_tokens, special_tokens=True)
            self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens) # converts tokens to tokenizer ids
            self.soft_prompts = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
                torch.empty(len(self.special_tokens), self.plm_embedding_dim)), requires_grad=True)


    def get_worst_tokens(self):
        pass

    def get_top_tokens(self):
        pass


    def generate_word_cloud(self, filename, top_tokens=50):
        pass

    def forward(self, x, prefixes=None, training=False):
        # x: np.array, pairs of (child, parent) (str)
        sentences = list(map(lambda pair: self.template.replace("[1]", pair[0]).replace("[2]", pair[1]), x))
        # print(sentences)
        if prefixes is not None:
            assert len(prefixes) == len(sentences)
            sentences = list(map(lambda x: x[1] + sentences[x[0]], enumerate(prefixes)))

        #print(sentences[0])

        tok = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"].to(device)

        special_token_ids = []
        attention_mask = 1 - tok.eq(self.tokenizer.pad_token_id).int()
        if self.special_tokens:
            for i, t in enumerate(self.special_tokens):
                special_token_ids.append(torch.nonzero(
                    tok == self.special_tokens_ids[i], as_tuple=True
                ))
                tok[special_token_ids[i]] = self.tokenizer.pad_token_id
               # tok[special_token_index_2] = self.tokenizer.pad_token_id

        if "roberta" in self.pretrained_model.lower():
            inputs_embeds = self.model.roberta.embeddings.word_embeddings(tok.int())
        else:
            inputs_embeds = self.model.bert.embeddings.word_embeddings(tok.int())

        # reparametrization trick on special tokens (soft prompts)
        if self.special_tokens:
            for i, t in enumerate(self.special_tokens):
                inputs_embeds[special_token_ids[i]] = inputs_embeds[special_token_ids[i]] * 0 + self.soft_prompts[i]
               # inputs_embeds[special_token_index_2] = inputs_embeds[special_token_index_2] * 0 + self.soft_prompts[1]


        logits = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
        ).logits

        mask_token_index = torch.nonzero(
            tok == self.tokenizer.mask_token_id, as_tuple=True
        )

        mask_logits = logits[mask_token_index]


        x = self.layernorm(mask_logits)
        x = self.dropout(x)
        x = self.vocab_extractor(x)

        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return torch.sigmoid(x.squeeze())

    def reg(self):
        return 0

    def save(self, filepath):
        torch.save(self.out_proj, os.path.join(filepath, "W_c.pt"))
        torch.save(self.vocab_extractor, os.path.join(filepath, "W_ve.pt"))
        if self.special_tokens:
            torch.save(self.soft_prompts, os.path.join(filepath, "soft_prompts.pt"))
        #torch.save(self.filter, os.path.join(filepath, "filter.pt"))

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.vocab_extractor = torch.load(os.path.join(filepath, "W_ve.pt"), map_location=device)
        model.out_proj = torch.load(os.path.join(filepath, "W_c.pt"), map_location=device)
        if os.path.exists(os.path.join(filepath, "soft_prompts.pt")):
            model.soft_prompts = torch.load(os.path.join(filepath, "soft_prompts.pt"), map_location=device)

        return model

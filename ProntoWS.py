import torch

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np

import os

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

class ProntoWS(torch.nn.Module):

    def __init__(self,
                 template="[1] [MASK] [2].",
                 special_tokens=None, #  ("[R1]", "[R2]")
                 pretrained_model="FacebookAI/bert-large",
                 init_token=None,
                 noise_scaling=1e-1,
                 l1_reg=1e-5):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model).to(device)
        self.template = template.replace("[MASK]", self.tokenizer.mask_token)
        self.l1_reg = l1_reg
        self.pretrained_model = pretrained_model
        if "roberta" in self.pretrained_model.lower():
            self.plm_num_embeddings = self.model.roberta.embeddings.word_embeddings.num_embeddings
            self.plm_embedding_dim = self.model.roberta.embeddings.word_embeddings.embedding_dim
        else:
            self.plm_num_embeddings = self.model.bert.embeddings.word_embeddings.num_embeddings
            self.plm_embedding_dim = self.model.bert.embeddings.word_embeddings.embedding_dim

        self.noise_scaling = noise_scaling
        for param in self.model.parameters():
            param.requires_grad = False

        self.verbalizer = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(1, self.plm_num_embeddings))
        )
        self.filter = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(1, self.plm_num_embeddings))
        )
        self.special_tokens = None

        # SPECIAL TOKENS
        if special_tokens:
            self.special_tokens = special_tokens
            self.tokenizer.add_tokens(self.special_tokens, special_tokens=True)
            self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens) # converts tokens to tokenizer ids
            self.soft_prompts = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
                torch.empty(len(self.special_tokens), self.plm_embedding_dim)), requires_grad=True)

        if init_token:
            init_token = self.tokenizer(init_token, add_special_tokens=False, padding=False)[
                "input_ids"
            ]
            with torch.no_grad():
                self.verbalizer[0, init_token] = 1.0

    def get_worst_tokens(self):
        return list(
            zip(
                self.tokenizer.batch_decode(
                    torch.argsort(self.verbalizer, descending=False).squeeze().cpu().detach().numpy()[:10].reshape(-1,
                                                                                                                  1)),
                torch.sort(torch.sigmoid(self.verbalizer), dim=-1, descending=False)[0][:, :10].tolist()[0]
            )
        )
    def get_top_tokens(self):
        return list(
            zip(
                self.tokenizer.batch_decode(torch.argsort(self.verbalizer, descending=True).squeeze().cpu().detach().numpy()[:10].reshape(-1, 1)),
                torch.sort(torch.sigmoid(self.verbalizer), dim=-1, descending=True)[0][:, :10].tolist()[0]
            )
        )

    def reg(self):
        return self.l1_reg * (
                torch.sum(torch.abs(torch.sigmoid(self.verbalizer))) +
                torch.sum(torch.abs(torch.sigmoid(self.filter)))
        )

    def generate_word_cloud(self, filename, top_tokens=30):
        # Convert tensor to numpy array
        tensor_array = (torch.sigmoid(self.verbalizer))[0].cpu().detach().numpy()

        # Get the indices of the highest values in the tensor
        top_indices = np.argsort(tensor_array)[::-1][:20]  # Adjust the number of tokens as needed

        # Create a dictionary of tokens and their importance scores
        token_importance = {self.tokenizer.decode(i): tensor_array[i] for i in top_indices}

        # Generate the word cloud
        wordcloud = WordCloud(
            width=450, height=450, background_color="white", min_font_size=10
        ).generate_from_frequencies(token_importance)

        # Display the word cloud
        plt.figure(figsize=(4, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        #plt.show()
        plt.savefig(f'{filename}.png')

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

        #if training:
        #    std = torch.std(mask_logits)
        #    noise = torch.randn_like(mask_logits) * std * self.noise_scaling
        #    mask_logits = mask_logits + noise


        mask_logits = (torch.exp(mask_logits) * torch.sigmoid(self.filter)) / torch.sum(torch.exp(mask_logits) * torch.sigmoid(self.filter), dim=-1).reshape(-1, 1)

        # Compute rescaled logits
        rescaled_logits = torch.sum(
            torch.sigmoid(self.verbalizer) * mask_logits,  # * torch.sigmoid(self.filter),
            dim=-1,
        )

        return rescaled_logits

    def save(self, filepath):
        torch.save(self.verbalizer, os.path.join(filepath, "verbalizer.pt"))
        torch.save(self.filter, os.path.join(filepath, "filter.pt"))
        if self.special_tokens:
            torch.save(self.soft_prompts, os.path.join(filepath, "soft_prompts.pt"))
        #torch.save(self.filter, os.path.join(filepath, "filter.pt"))

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.verbalizer = torch.load(os.path.join(filepath, "verbalizer.pt"), map_location=device)
        model.filter = torch.load(os.path.join(filepath, "filter.pt"), map_location=device)
        if os.path.exists(os.path.join(filepath, "soft_prompts.pt")):
            model.soft_prompts = torch.load(os.path.join(filepath, "soft_prompts.pt"), map_location=device)

        return model

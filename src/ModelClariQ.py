from collections import defaultdict
from math import isnan
from sklearn.metrics import ndcg_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
import math
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ClariQDataset import ClariQDataset

from IPython import embed

class QuestionGenGPT2(pl.LightningModule):
    def __init__(self, hparams):
        super(QuestionGenGPT2, self).__init__()
        self.hparams = hparams

        SPECIAL_TOKENS = {'pad_token': '<pad>',
                          'sep_token': '<sep>',
                          'bos_token': '<bos>',
                          'eos_token': '<eos>'}

        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return outputs


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, correct_bias=True)
        return optimizer

    def train_dataloader(self):
        dataset = ClariQDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='train'
                        )

        sampler = None
        self.train_dataloader_object = DataLoader(
                                    dataset, batch_size=self.hparams.data_loader_bs,
                                    shuffle=(sampler is None),
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    collate_fn=QuestionGenGPT2.collate_fn
                                    )
        return self.train_dataloader_object

    def val_dataloader(self):
        dataset = ClariQDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='dev'
                            )

        sampler = None
        self.val_dataloader_object = DataLoader(
                                    dataset, batch_size=self.hparams.val_data_loader_bs,
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    collate_fn=QuestionGenGPT2.collate_fn
                                    )
        return self.val_dataloader_object

    def test_dataloader(self):
        dataset = ClariQDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='test'
                            )

        sampler = None
        self.test_dataloader_object = DataLoader(
                                    dataset, batch_size=1, # 1 cuz sampling
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    # collate_fn=QuestionGenGPT2.collate_fn_test
                                    )
        return self.test_dataloader_object

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self.forward(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

        loss = output[0]
        if self.logger:
            self.logger.log_metrics({'train_loss': loss.item()})

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self.forward(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

        loss = output[0]
        if self.logger:
            self.logger.log_metrics({'val_loss': loss.item()})

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        """ 
        outputs: dict of outputs of validation_step (or validation_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['probs'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print(f"\nDEV:: avg-LOSS: {avg_loss} ||")

        return {'val_epoch_loss': avg_loss}



    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([x['input_ids'] for x in batch])
        token_type_ids = torch.stack([x['token_type_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])

        lm_label = torch.stack([x['lm_label'] for x in batch])

        return (input_ids, attention_mask, token_type_ids, lm_label)
    
    def test_step(self, batch, batch_nb):
        # batch size is 1 cuz of sampling

        facets = batch['facets'][0]
        history = batch['history'][0]
        out_ids = self.sample_sequence(facets, history)
        out_test = self.tokenizer.decode(out_ids)

        with open('out/' + self.hparams.run_name + '_' + self.hparams.test_ckp.split('=')[1] + '_' + \
                str(self.hparams.temperature) + '_' + \
                str(self.hparams.top_k) + '_' + \
                str(self.hparams.top_p) + '.out', 'a') as fout:
            fout.write(facets + '\t' + history + '\t' + out_test + '\n')
        
    
    # probs not needed with the current setup
    # def test_epoch_end(self, outputs):
        # """ 
        # outputs: dict of outputs of test_step (or test_step_end in dp/ddp2)
        # outputs['loss'] --> losses of all the batches
        # outputs['probs'] --> scores for each example
        # outputs['idxs'] --> indexes in Dataset to connect with scores
        # """

        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

    def sample_sequence(self, facets, history, current_output=None, question=None):
        temperature = self.hparams.temperature
        top_k = self.hparams.top_k
        top_p = self.hparams.top_p

        if current_output is None:
            current_output = []

        for i in range(self.hparams.max_output_len):
            instance = ClariQDataset.build_input_from_segments(facets, history, current_output, self.tokenizer,
                                                        with_eos=False, without_facets=self.hparams.without_facets)
            input_ids = torch.LongTensor([instance['input_ids']]).to('cuda')
            token_type_ids = torch.LongTensor([instance['token_type_ids']]).to('cuda')
            logits = self.model(input_ids, token_type_ids=token_type_ids) # attention_mask? I guess no need cuz it's not padded!
            if isinstance(logits, tuple):
                logits = logits[0]

            # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
            logits = logits[0, -1, :] / temperature
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            special_token_ids = self.tokenizer.all_special_ids
            if i < self.hparams.min_output_len and prev.item() in special_token_ids:
                n_special = 0
                while prev.item() in special_token_ids:
                    n_special += 1
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    if n_special > 100:
                        warnings.warn("Warning: model generating special token 100 times in a row.")
                        break
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_token_ids:
                break
            current_output.append(prev.item()) #self.tokenizer.decode(prev.item())

        return current_output

    # from HuggingFace
    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
        """

        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits


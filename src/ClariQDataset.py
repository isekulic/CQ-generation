import argparse
import json
import pandas as pd
import pickle
import os
import torch

from collections import defaultdict
from IPython import embed
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ClariQDataset(Dataset):
    def __init__(self, tokenizer, args, mode='dev'):
        self.tokenizer = tokenizer
        self.data_dir = args.data_dir
        self.mode = mode
        self.max_seq_len = args.max_seq_len
        self.hparams = args

        if self.hparams.use_faceted_data:
            self.df = pd.read_csv(args.my_faceted_data +'.'+ mode, sep='\t')
        else:
            self.df = pd.read_csv(os.path.join(self.data_dir,'ClariQ-'+mode+'.tsv'), sep='\t')
            print(len(self.df))
            self.df = self.df.dropna()
            print(len(self.df))

            self.facets = self.df[['topic_id', 'facet_desc']]

    def __len__(self):

        return len(self.df)
    
    def __getitem__(self, idx):
        # do not pad & create batches, just build_build_input_from_segments
        if self.mode == 'test':
            return self.test_mode_to_tensor(idx)

        return self.example_to_tensor(idx)

    def example_to_tensor(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        cq = sample.question
        facet = sample.facet_desc

        instance = self.build_input_from_segments(facet, q, cq, self.tokenizer, 
                                            lm_labels=True, with_eos=True,
                                            without_facets=self.hparams.without_facets)

        input_seq = instance['input_ids']
        segments = instance['token_type_ids']
        mask = instance['attention_mask']
        target = instance['lm_label']

        if len(input_seq) > self.hparams.max_seq_len:
            input_seq = input_seq[:self.hparams.max_seq_len]
            target = target[:self.hparams.max_seq_len]
            segments = segments[:self.hparams.max_seq_len]
            mask = mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(input_seq)
            input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            target.extend([-100] * pad_num)
            segments.extend([self.tokenizer.pad_token_id] * pad_num)
            mask.extend([0] * pad_num)

        ret = {}
        ret['input_ids'] = torch.LongTensor(input_seq)
        ret['lm_label'] = torch.LongTensor(target)
        # dialogue state embeddings
        ret['token_type_ids'] = torch.LongTensor(segments)
        # attention mask to distinguish padding and real text
        ret['attention_mask'] = torch.LongTensor(mask)

        return ret



    def test_mode_to_tensor(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        facet = sample.facet_desc
        # facets = ' '.join(self.facets[self.facets['topic_id'] == sample.topic_id].facet_desc.unique())

        return {'facets': facet, 'history': q}


    @staticmethod
    def build_input_from_segments(facets, history, question, tokenizer, lm_labels=False, with_eos=True, without_facets=False):
        """ Build an input sequence from facets descriptions, conv. history, and the clarifying question."""
        segment_id = 0

        input_seq, segments = [], []
        if without_facets: # used for query-only baseline
            input_seq += tokenizer.encode(history)
            segments += [segment_id] * (len(input_seq) - len(segments))
            segment_id += 1
            # input_seq.append(tokenizer.sep_token_id) # here, not in target
            input_seq.append(tokenizer.bos_token_id)

        else: # add everything <facet terms> [SEP] <query> [bos] ...
            # adding facet descriptions
            input_seq += tokenizer.encode(facets)
            segments += [segment_id] * len(input_seq)
            segment_id += 1

            # adding conv history (or just initial query for first turn)
            input_seq.append(tokenizer.sep_token_id)
            input_seq += tokenizer.encode(history)
            segments += [segment_id] * (len(input_seq) - len(segments))
            segment_id += 1
            # input_seq.append(tokenizer.sep_token_id) # here, not in target

            input_seq.append(tokenizer.bos_token_id)

        target_raw = question

        if isinstance(target_raw, str):
            tmp = tokenizer.encode(target_raw)
        else:
            tmp = target_raw
        if with_eos:
            tmp.append(tokenizer.eos_token_id)

        if lm_labels:
            target = [-100] * len(input_seq)
            target += tmp 
        input_seq += tmp

        segments += [segment_id] * (len(input_seq) - len(segments))
        segment_id += 1
        mask = [1] * len(input_seq)

        instance = {}
        instance['input_ids'] = input_seq
        instance['lm_label'] = target if lm_labels else []
        # dialogue state embeddings
        instance['token_type_ids'] = segments
        # attention mask to distinguish padding and real text
        instance['attention_mask'] = mask

        return instance

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument('--data_dir', type=str, default='../data/')
        parser.add_argument('--mode', type=str, default='dev')
        parser.add_argument('--max_seq_len', type=int, default=512)
        return parser

if __name__ == "__main__":
    # used only for testing purposes -- code should be ran from run.py
    main_arg_parser = argparse.ArgumentParser(description="ClariQ dataset")
    parser = ClariQDataset.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()

    SPECIAL_TOKENS = {'pad_token': '<pad>',
                      'sep_token': '<sep>',
                      'bos_token': '<bos>',
                      'eos_token': '<eos>'}

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    cd = ClariQDataset(tokenizer, args)

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))

    X, Y, segments = [], [], []
    for i in range(5):
        xs = cd[i]
        X.append(xs['input_seq'])
        Y.append(xs['target'])
        segments.append(xs['token_type_ids'])

    X = torch.stack(X).to('cuda')
    Y = torch.stack(Y).to('cuda')
    segments = torch.stack(segments).to('cuda')

    model.to('cuda')

    out = model(X, token_type_ids=segments,
                labels=Y)
    loss = out[0]
    loss.backward()
    embed()


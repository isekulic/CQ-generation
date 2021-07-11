import argparse
import os
import pytorch_lightning as pl
import torch

from ClariQDataset import ClariQDataset
from ModelClariQ import QuestionGenGPT2
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

seed = 42
seed_everything(seed)

def main(hparams):

    model = QuestionGenGPT2(hparams)
    if hparams.test_mode and hparams.test_ckp:
        checkpoint = torch.load(hparams.test_ckp)
        model.load_state_dict(checkpoint['state_dict'])
    print(model)

    loggers = []
    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger("tb_logs", name=f"{hparams.run_name}",
                                    version=hparams.slurm_job_id)
        loggers.append(tb_logger)

    checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), 'checkpoints'),
                save_top_k=1,
                verbose=True,
                monitor='val_epoch_loss',
                mode='min',
                prefix=''
                )

    trainer = pl.Trainer(
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            # distributed_backend=hparams.distributed_backend,
            # control the effective batch size with this param
            accumulate_grad_batches=hparams.trainer_batch_size,
            # Training will stop if max_steps or max_epochs have reached (earliest).
            max_epochs=hparams.epochs,
            # max_steps=hparams.num_training_steps, 
            logger=loggers,
            checkpoint_callback=checkpoint_callback,
            # progress_bar_callback=False,
            # progress_bar_refresh_rate=0,
            # use_amp=True --> use 16bit precision
            # val_check_interval=20, # val 4 times during 1 train epoch
            # val_check_interval=hparams.val_check_interval, # val every N steps
            num_sanity_val_steps=5,
            # fast_dev_run=True
        )


    if hparams.test_mode:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__=='__main__':

    # print(os.environ['SLURM_NODELIST'])
    parser = argparse.ArgumentParser(description='CQ-gen')
    # MODEL SPECIFIC
    parser.add_argument("--model_name", type=str, default='distilgpt2',
                        help="Model name")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum number of wordpieces of the sequence")
    parser.add_argument("--lr", type=float, default=6.25e-5, # from HuggingFace
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Num subprocesses for DataLoader")
    parser.add_argument('--run_name', type=str, default='initial')
    parser.add_argument('--test_ckp', type=str, default='checkpoints/epoch=4_v3.ckpt')
    parser.add_argument('--use_faceted_data', type=int, default=0)
    parser.add_argument('--without_facets', type=int, default=0)
    parser.add_argument('--my_faceted_data', type=str, default='../data/ClariQ-FKw.tsv')

    # SAMPLING HPARAMS
    parser.add_argument("--max_output_len", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_output_len", type=int, default=3, help="Minimum length of the output utterances")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")


    # EXPERIMENT SPECIFIC
    parser.add_argument("--data_dir", type=str, default='../data/',)
    # effective batch size will be: 
    # trainer_batch_size * data_loader_bs
    parser.add_argument("--trainer_batch_size", type=int, default=8,
                        help='Batch size for Trainer. Accumulates grads every k batches')
    parser.add_argument("--data_loader_bs", type=int, default=4,
                        help='Batch size for DataLoader object')
    parser.add_argument("--val_data_loader_bs", type=int, default=0,
                        help='Batch size for validation data loader. If not specified,\
                        --data_loader_bs is used.')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--slurm_job_id", type=int, default=1)
    parser.add_argument("--use_tensorboard", type=int, default=1,
                        help='Use TensorBoard logger (default in PL)')
    parser.add_argument("--test_mode", type=int, default=0)

    # Distributed training
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Num nodes allocated by SLURM")
    parser.add_argument("--distributed_backend", type=str, default='ddp',
                        help="Use distributed backend: dp/ddp/ddp2")

    hparams = parser.parse_args()
    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs
    print(hparams)
    main(hparams)


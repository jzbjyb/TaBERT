from argparse import ArgumentParser
from pathlib import Path
import contextlib
import os
import sys
import random
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import DataLoader
import numpy as np
from fairseq import optim
from fairseq.optim import lr_scheduler
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule
from fairseq.options import eval_str_list
from table_bert import TableBertModel
from table_bert.utils import compute_mrr
from table_bert.ranker import RankDataset, Ranker


def build_optimizer(model, args):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.build_optimizer(args, params)
    scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)
    scheduler.step_update(0)
    return optimizer, scheduler


def main():
    parser = ArgumentParser()
    parser.add_argument('--sample_file', type=Path, required=True)
    parser.add_argument('--db_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, default=True)
    parser.add_argument('--load_from', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--group_by', type=int, default=10)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument("--lr-scheduler", type=str, default='polynomial_decay', help='Learning rate scheduler')
    parser.add_argument("--optimizer", type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--clip-norm', default=0., type=float, help='clip gradient')
    parser.add_argument('--lr', '--learning-rate', default='0.00005', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    FairseqAdam.add_args(parser)
    PolynomialDecaySchedule.add_args(parser)

    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build model
    model = Ranker(TableBertModel.from_pretrained(str(args.model_path)))
    if args.load_from:
        print('load from {}'.format(args.load_from))
        state_dict = torch.load(args.load_from, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    model = model.to(device)
    model.train() if args.finetune else model.eval()

    # build dataset
    rankdataset = RankDataset(args.sample_file, args.db_file, group_by=args.group_by, tokenizer=model.tokenizer)
    dataloader = DataLoader(rankdataset, batch_size=args.batch_size, num_workers=0, shuffle=args.finetune,
                            collate_fn=partial(RankDataset.collate, model=model.model))

    # build optimizer
    args.total_num_updates = len(rankdataset) * args.max_epoch // args.batch_size
    args.warmup_updates = int(args.total_num_updates * 0.1)
    print('#samples in the data {}, total_num_updates {}, warmup_updates {}'.format(
        len(rankdataset), args.total_num_updates, args.warmup_updates))
    optimizer, scheduler = build_optimizer(model, args)

    num_updates = 0
    with contextlib.ExitStack() if args.finetune else torch.no_grad():
        for e in range(args.max_epoch if args.finetune else 1):
            all_scores = []
            all_labels = []
            with tqdm(total=len(dataloader), desc='Epoch {}'.format(e), file=sys.stdout, miniters=10) as pbar:
                for tensor_dict, labels in dataloader:
                    loss, scores = model(tensor_dict, labels=labels)
                    all_scores.extend(scores.detach().cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    if args.finetune:
                        optimizer.zero_grad()
                        optimizer.backward(loss)
                        optimizer.clip_grad_norm(args.clip_norm)
                        optimizer.step()
                        num_updates += 1
                        scheduler.step_update(num_updates)
                    pbar.update(1)
                    pbar.set_postfix_str('loss: {}'.format(loss.item()))
            if args.finetune and args.output_file:  # save model
                print('save model for epoch {}'.format(e + 1))
                os.makedirs(args.output_file, exist_ok=True)
                torch.save(model.state_dict(), str(args.output_file / 'model_epoch{}.bin'.format(e + 1)))

    if not args.finetune:
        mrrs = []
        with open(args.output_file, 'w') as fout:
            for i in range(0, len(all_labels), args.group_by):
                ls = all_labels[i:i + args.group_by]
                ss = all_scores[i:i + args.group_by]
                mrrs.append(compute_mrr(ss, ls))
                for s in ss:
                    fout.write('{}\n'.format(s))
        print('MRR {}'.format(np.mean(mrrs)))


if __name__ == '__main__':
    main()

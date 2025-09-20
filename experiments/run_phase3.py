"""
Experiment launcher for Phase 3 (knowledge distillation).

This script loads phase1 + phase2 ckpts, freezes auxiliary models,
and calls the trainer for Phase 3, preserving original training logic
and checkpoint naming.
Author: Tian Yuxuan
Date: 2025-08-21
"""

import random
import argparse
from data.dataset import *
from models.encoder import *
from utils.dataloader import *
from data.subgraph_builder import *
from models.loss.contrast_loss import *
from utils.metapath_config import view_metas
from data.splits import split_ids_from_list_pkl
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainers.phase3_distill import run_phase3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_kd', type=int, default=50)
    parser.add_argument('--epoch_test', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default="teacher")
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default="./cache")
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_students', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--mimic3_path', type=str, default="", help="Root path to MIMIC-III dataset (e.g., .../mimic3/hosp)")
    parser.add_argument('--mimic4_path', type=str, default="", help="Root path to MIMIC-IV dataset (e.g., .../mimic4/hosp)")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu")

    # load dataset and tokenizers
    try:
        if args.dataset == 'mimic4':
            task_dataset = load_dataset(args.dataset, root=args.mimic4_path, task_fn=diag_prediction_mimic4_fn, dev=False)
        else:
            task_dataset = load_dataset(args.dataset, root=args.mimic3_path, task_fn=diag_prediction_mimic3_fn, dev=False)
    except FileNotFoundError as e:
        print(f"File path error: {str(e)}")
        return

    Tokenizers = get_init_tokenizers(task_dataset)
    label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))

    # prepare dataloaders used by Phase3 (we reuse aux train dataloader from Phase2)
    cache_path = os.path.join(args.data_path, args.dataset)
    train_aux, val_set, test_set = prepare_aux_dataset(
        id_split_dir='cache/splits',
        graph_pkl_path=os.path.join(cache_path, 'patient_disease.pkl'),
        seq_pkl_path=os.path.join(cache_path, 'seqdata.pkl'),
        drug_pkl_path=os.path.join(cache_path, 'drug_disease.pkl'),
        procedure_pkl_path=os.path.join(cache_path, 'procedures_disease.pkl')
    )
    train_aux_loader, val_aux_loader, test_aux_loader = view_dataloader(train_aux, val_set, test_set, args.batch_size)

    # checkpoint paths
    ckpt_phase1 = f'./ckpt/phase1/phase1_{args.model}_{args.dataset}_{args.seed}.ckpt'
    ckpt_phase2_pred = f'./ckpt/phase2/phase2_pred_{args.model}_{args.dataset}_{args.seed}.ckpt'
    ckpt_phase3 = f'./ckpt/phase3/phase3_{args.model}_{args.dataset}_{args.seed}.ckpt'
    os.makedirs(os.path.dirname(ckpt_phase3), exist_ok=True)

    # If final ckpt exists, skip
    if os.path.exists(ckpt_phase3):
        print(f"Detected existing Phase 3 ckpt {ckpt_phase3}, skip Phase 3 training.")
        return

    # Instantiate main and auxiliary models
    main_model = Teacher(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device,
                         graph_meta=view_metas['patient_disease']).to(device)
    aux_model1 = Student_med(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device, num_heads=2, num_layers=1).to(device)
    aux_model2 = Student_proc(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device, num_heads=2, num_layers=1).to(device)

    # load phase1 ckpt and phase2 pred ckpt into models
    phase1_ckpt = torch.load(ckpt_phase1, map_location=device)
    phase2_pred_ckpt = torch.load(ckpt_phase2_pred, map_location=device)
    main_model.load_state_dict(phase1_ckpt)
    aux_model1.load_state_dict(phase2_pred_ckpt['aux_model1'])
    aux_model2.load_state_dict(phase2_pred_ckpt['aux_model2'])

    # freeze auxiliary models
    aux_model1.eval()
    aux_model2.eval()
    for p in aux_model1.parameters():
        p.requires_grad = False
    for p in aux_model2.parameters():
        p.requires_grad = False

    # optimizer / scheduler for main model training
    optimizer = torch.optim.AdamW(list(main_model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    contrast = ContrastiveLearner(temperature=0.07)

    # call Phase3 trainer (it will save ckpt based on metric_code)
    run_phase3(
        train_aux_loader=train_aux_loader,
        val_aux_loader=val_aux_loader,
        test_aux_loader=test_aux_loader,
        main_model=main_model,
        aux_model1=aux_model1,
        aux_model2=aux_model2,
        label_tokenizer=label_tokenizer,
        args=args,
        device=device,
        ckpt_dir='./ckpt/phase3'
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

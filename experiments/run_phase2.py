"""
Experiment launcher for Phase 2 (auxiliary views contrastive learning).

This script prepares auxiliary view dataloaders (prepare_aux_dataset + view_dataloader),
loads Phase1 ckpt into main_model, instantiates aux_model1/aux_model2, then calls the trainer.
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

from trainers.phase2_collab import run_phase2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_view', type=int, default=20)
    parser.add_argument('--epoch_test', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default="teacher")
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default="./cache")
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--mimic3_path', type=str, default="", help="Root path to MIMIC-III dataset (e.g., .../mimic3/hosp)")
    parser.add_argument('--mimic4_path', type=str, default="", help="Root path to MIMIC-IV dataset (e.g., .../mimic4/hosp)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu")

    # load dataset and tokenizers (assume same fileroot mapping available)
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

    # prepare auxiliary dataloaders
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
    ckpt_phase2_repr = f'./ckpt/phase2/phase2_repr_{args.model}_{args.dataset}_{args.seed}.ckpt'
    ckpt_phase2_pred = f'./ckpt/phase2/phase2_pred_{args.model}_{args.dataset}_{args.seed}.ckpt'
    os.makedirs(os.path.dirname(ckpt_phase2_repr), exist_ok=True)

    # skip training if repr ckpt exists (original logic checks repr ckpt; preserve that)
    if os.path.exists(ckpt_phase2_repr):
        print(f"Detected existing Phase 2 repr ckpt {ckpt_phase2_repr}, skip Phase 2 training.")
    else:
        # load teacher from phase1 checkpoint
        main_model = Teacher(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device, graph_meta=view_metas['patient_disease'])
        best_model = torch.load(ckpt_phase1, map_location=device)
        main_model.load_state_dict(best_model)
        main_model = main_model.to(device)
        main_model.eval()
        for param in main_model.parameters():
            param.requires_grad = False

        # instantiate auxiliary students
        aux_model1 = Student_med(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device, num_heads=2, num_layers=1).to(device)
        aux_model2 = Student_proc(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device, num_heads=2, num_layers=1).to(device)

        # call trainer
        run_phase2(
            train_aux_loader=train_aux_loader,
            val_aux_loader=val_aux_loader,
            test_aux_loader=test_aux_loader,
            main_model=main_model,
            aux_model1=aux_model1,
            aux_model2=aux_model2,
            label_tokenizer=label_tokenizer,
            args=args,
            device=device,
            ckpt_dir='./ckpt/phase2'
        )

    # after Phase 2 you can optionally load the best prediction ckpt for downstream
    if os.path.exists(ckpt_phase2_pred):
        phase2_ckpt_pred = torch.load(ckpt_phase2_pred, map_location=device)
        # caller/Phase3 script will load these into aux_model1/aux_model2 as needed
        print(f"Phase 2 best pred ckpt saved at {ckpt_phase2_pred}")
    else:
        print("Phase 2 pred ckpt not found after training.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

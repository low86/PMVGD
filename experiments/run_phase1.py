"""
Experiment launcher for Phase 1.
This script prepares dataset/tokenizers/cache/splits and then calls trainers.phase1_pretrain.run_phase1.
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

from trainers.phase1_pretrain import run_phase1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_main', type=int, default=130, help='Number of epochs to train phase 1.')
    parser.add_argument('--epoch_test', type=int, default=10, help='Number of epochs to test.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--model', type=str, default="teacher", help='teacher')
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--devm', type=bool, default=False, help='develop mode')
    parser.add_argument('--test_epoch', type=int, default=10, help='test epoch')
    parser.add_argument('--data_path', type=str, default="./cache", help='data path')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--mimic3_path', type=str, default=""
                        , help="Root path to MIMIC-III dataset (e.g., .../mimic3/hosp)")
    parser.add_argument('--mimic4_path', type=str, default=""
                        , help="Root path to MIMIC-IV dataset (e.g., .../mimic4/hosp)")
    args = parser.parse_args()

    # environment & seeds
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f'{args.dataset}--{args.model}')

    # device
    cudaid = f"cuda:{args.dev}"
    device = torch.device(cudaid if torch.cuda.is_available() else "cpu")

    # load dataset
    try:
        if args.dataset == 'mimic4':
            task_dataset = load_dataset(args.dataset, root=args.mimic4_path, task_fn=diag_prediction_mimic4_fn, dev=args.devm)
        elif args.dataset == 'mimic3':
            task_dataset = load_dataset(args.dataset, root=args.mimic3_path, task_fn=diag_prediction_mimic3_fn, dev=args.devm)

    except FileNotFoundError as e:
        print(f"File path error: {str(e)}")
        print(f"Please check the dataset root path.")
        return

    Tokenizers = get_init_tokenizers(task_dataset)
    label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))

    # optional pandarallel (as in original)
    try:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=16, progress_bar=True, use_memory_fs=False)
        print("Pandarallel enabled.")
    except Exception:
        print("Pandarallel not available; proceeding without it.")

    # build cache & graph if missing
    cache_path = os.path.join(args.data_path, args.dataset)
    test_path = os.path.join(cache_path, 'patient_disease.pkl')
    if not os.path.exists(test_path):
        builder = GraphBuilder(dataset=task_dataset, tokenizer=Tokenizers, dim=32, device=device, cache_dir=cache_path)
        builder.build_and_save_all()

    # split ids (if pkl exists)
    if os.path.exists(test_path):
        split_ids_from_list_pkl(pkl_path=test_path, save_dir='cache/splits', train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, seed=args.seed)

    # prepare dataloaders
    trainset, validset, testset = split_dataset(id_split_dir='cache/splits', graph_pkl_path=test_path,
                                                seq_pkl_path=os.path.join(cache_path, 'seqdata.pkl'))
    train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)

    # build teacher model
    model = Teacher(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device,
                    graph_meta=view_metas['patient_disease'])

    # checkpoint path
    ckptpath = f'./ckpt/phase1/phase1_{args.model}_{args.dataset}_{args.seed}.ckpt'
    os.makedirs(os.path.dirname(ckptpath), exist_ok=True)

    # call trainer
    run_phase1(train_loader, val_loader, test_loader, model, label_tokenizer, args, device, ckptpath)

    # post-processing: same behavior as original main
    if args.model == 'teacher':
        # free and re-instantiate as in original script
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        device_cpu = torch.device('cpu')
        model = Teacher(Tokenizers, args.hidden_size, len(task_dataset.get_all_tokens('conditions')), device_cpu,
                        graph_meta=view_metas['patient_disease'])

    # load best model and final test
    best_model = torch.load(ckptpath, map_location=device)
    model.load_state_dict(best_model)
    model = model.to(device)
    y_true, y_prob = test_phase_one(test_loader, model, label_tokenizer, device)
    print(f"Code Level Metrics: {code_level(y_true, y_prob)}")
    print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")


if __name__ == '__main__':
    # Windows multiprocessing fix
    import multiprocessing
    multiprocessing.freeze_support()
    main()
#python -m experiments.run_phase1 --dataset mimic3 --model teacher --epoch_main 130 --mimic3_path "C:\Users\pc\Desktop\tyx\demo\mimic3\hosp"


"""
Test code for PMVGD on MIMIC-III / MIMIC-IV datasets.
Author: Tian Yuxuan
Date: 2025-08-09
"""
import random
import argparse
from data.dataset import *
from utils.dataloader import *

from models.encoder import *
from data.subgraph_builder import *
from utils.metapath_config import view_metas
from data.splits import split_ids_from_list_pkl
def run_test(args):
    """
    Evaluate a trained PMVGD teacher model on the chosen dataset.
    """
    fileroot = {
        'mimic3': 'The MIMIC-III dataset is located at this path',
        'mimic4': 'The MIMIC-IV dataset is located at this path',
    }

    # Set random seeds
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu")
    print(f"Testing {args.model} on {args.dataset} | Device: {device}")

    # Load dataset
    try:
        if args.dataset == 'mimic4':
            task_dataset = load_dataset(args.dataset,
                                        root=fileroot[args.dataset],
                                        task_fn=diag_prediction_mimic4_fn,
                                        dev=args.devm)
        elif args.dataset == 'mimic3':
            task_dataset = load_dataset(args.dataset,
                                        root=fileroot[args.dataset],
                                        task_fn=diag_prediction_mimic3_fn,
                                        dev=args.devm)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    except FileNotFoundError as e:
        print(f"[Error] {str(e)}")
        print(f"Please check dataset path: {fileroot[args.dataset]}")
        return

    # Tokenizers
    tokenizers = get_init_tokenizers(task_dataset)
    label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))

    # Graph cache
    cache_path = os.path.join(args.data_path, args.dataset)
    test_path = os.path.join(cache_path, 'patient_disease.pkl')
    if not os.path.exists(test_path):
        builder = GraphBuilder(dataset=task_dataset,
                               tokenizer=tokenizers,
                               dim=args.hidden_size,
                               device=device,
                               cache_dir=cache_path)
        builder.build_and_save_all()

    # Train/Val/Test splits
    split_ids_from_list_pkl(
        pkl_path=test_path,
        save_dir='cache/splits',
        train_ratio=0.75, val_ratio=0.1, test_ratio=0.15,
        seed=args.seed
    )
    trainset, validset, testset = split_dataset(
        id_split_dir='cache/splits',
        graph_pkl_path=test_path,
        seq_pkl_path=os.path.join(cache_path, 'seqdata.pkl')
    )
    _, _, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)

    # Model
    model = Teacher(tokenizers,
                    args.hidden_size,
                    len(task_dataset.get_all_tokens('conditions')),
                    device,
                    graph_meta=view_metas['patient_disease'])

    ckptpath = f'./ckpt/phase1/trained_{args.model}_{args.dataset}_{args.seed}.ckpt'
    if not os.path.exists(ckptpath):
        print(f"[Error] Model checkpoint not found: {ckptpath}")
        return

    # Load model checkpoint
    model.load_state_dict(torch.load(ckptpath, map_location=device))
    model = model.to(device)
    print(f"[Info] Loaded trained model from {ckptpath}")

    # Evaluate
    y_true, y_prob = test_phase_one(test_loader, model, label_tokenizer, device)
    print(f"Code Level Metrics: {code_level(y_true, y_prob)}")
    print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--devm', type=bool, default=False, help='develop mode')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--data_path', type=str, default="./cache", help='data path')
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--model', type=str, default="PMVGD", help='model name')
    args = parser.parse_args()

    run_test(args)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
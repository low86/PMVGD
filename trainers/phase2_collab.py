# trainers/phase2_collab.py
"""
Phase 2 trainer: Multi-view collaborative learning (auxiliary views).

This function expects prepared auxiliary dataloaders and the main_model + aux_models
already instantiated by the experiment. It uses your project's helper
train_phase_two and test_phase_two functions to compute losses and testing.
Author: Tian Yuxuan
Date: 2025-08-24
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


def run_phase2(train_aux_loader,
               val_aux_loader,
               test_aux_loader,
               main_model,
               aux_model1,
               aux_model2,
               label_tokenizer,
               args,
               device,
               ckpt_dir='./ckpt/phase2'):
    """
    Train auxiliary view encoders (Phase 2).

    Args:
        train_aux_loader, val_aux_loader, test_aux_loader: prepared DataLoader objects
        main_model: pretrained main model (teacher) instance
        aux_model1, aux_model2: auxiliary student model instances
        label_tokenizer: tokenizer instance for labels
        args: argparse.Namespace containing epoch_view, lr, epoch_test, model, dataset, seed
        device: torch.device
        ckpt_dir: directory to save phase2 ckpts
    Returns:
        repr_ckpt_path, pred_ckpt_path
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    repr_ckpt = os.path.join(ckpt_dir, f'phase2_repr_{args.model}_{args.dataset}_{args.seed}.ckpt')
    pred_ckpt = os.path.join(ckpt_dir, f'phase2_pred_{args.model}_{args.dataset}_{args.seed}.ckpt')

    optimizer_aux = torch.optim.AdamW(list(aux_model1.parameters()) + list(aux_model2.parameters()), lr=args.lr)
    scheduler_aux = CosineAnnealingLR(optimizer_aux, T_max=10, eta_min=1e-5)
    contrast = ContrastiveLearner(temperature=0.07)

    best_repr_loss = float("inf")
    best_pred_loss = float("inf")
    best_model_repr = None
    best_model_pred = None

    pbar_aux = tqdm(range(args.epoch_view), desc="Training Auxiliary Views")
    for epoch in pbar_aux:
        # train_phase_two returns (total_contrast_loss, total_pred_loss) as in your original code
        total_contrast_loss, total_pred_loss = train_phase_two(
            train_aux_loader, main_model, aux_model1, aux_model2, label_tokenizer, contrast, optimizer_aux, device
        )

        # step scheduler
        scheduler_aux.step()

        # compute averages consistent with your original code
        avg_contrast_loss = total_contrast_loss / len(train_aux_loader)
        # total_pred_loss may be a tensor, keep original .item() usage if available
        try:
            avg_pred_loss = total_pred_loss.item() / len(train_aux_loader)
        except Exception:
            avg_pred_loss = total_pred_loss / len(train_aux_loader)

        pbar_aux.set_description(
            f"Phase 2  | Epoch {epoch + 1}/{args.epoch_view} | Repr: {avg_contrast_loss:.4f} | Pred: {avg_pred_loss:.4f}"
        )

        # save best representation checkpoint immediately when improved
        if avg_contrast_loss < best_repr_loss:
            best_repr_loss = avg_contrast_loss
            best_model_repr = {
                'aux_model1': aux_model1.state_dict(),
                'aux_model2': aux_model2.state_dict(),
            }
            torch.save(best_model_repr, repr_ckpt)

        # save best prediction checkpoint immediately when improved
        if avg_pred_loss < best_pred_loss:
            best_pred_loss = avg_pred_loss
            best_model_pred = {
                'aux_model1': aux_model1.state_dict(),
                'aux_model2': aux_model2.state_dict(),
            }
            torch.save(best_model_pred, pred_ckpt)

        # periodic test every 5 epochs (as in your original script)
        if epoch % 5 == 0:
            if best_model_pred is not None:
                aux_model1.load_state_dict(best_model_pred['aux_model1'])
                aux_model1.to(device)
                y_true, y_prob = test_phase_two(test_aux_loader, aux_model1, main_model, label_tokenizer, device, show_progress=False)
                print(f"Test-{epoch // 5} Code Level Metrics: {code_level(y_true, y_prob)}")
                print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")
            else:
                print(f"Phase 2 test skipped at epoch {epoch}: no best pred ckpt yet.")

    # final evaluation using best prediction ckpt if exists
    if best_model_pred is not None:
        aux_model1.load_state_dict(best_model_pred['aux_model1'])
        aux_model1.to(device)
        y_true, y_prob = test_phase_two(test_aux_loader, aux_model1, main_model, label_tokenizer, device)
        print("Final Phase 2 evaluation (using best pred ckpt):")
        print(f"Code Level Metrics: {code_level(y_true, y_prob)}")
        print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")
    else:
        print("Warning: no best prediction checkpoint was saved during Phase 2.")

    return repr_ckpt, pred_ckpt

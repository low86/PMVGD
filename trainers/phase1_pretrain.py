"""
Phase 1 trainer: Teacher pretraining.

Receives prepared dataloaders and a model instance from experiments/*.py,
runs the training loop, saves best checkpoint by validation loss, and
periodically evaluates using test_phase_one (exactly as in your original script).
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

def run_phase1(train_loader,
               val_loader,
               test_loader,
               model,
               label_tokenizer,
               args,
               device,
               ckptpath):
    """
    Train the teacher model (Phase 1).

    Args:
        train_loader, val_loader, test_loader: prepared DataLoader objects
        model: Teacher model instance (already instantiated by experiments)
        label_tokenizer: label tokenizer instance
        args: argparse.Namespace with fields:
              epoch_main, epoch_test, lr, model, dataset, seed, ...
        device: torch.device
        ckptpath: path to save best checkpoint (string)
    Returns:
        ckptpath (string): saved checkpoint path
    """
    os.makedirs(os.path.dirname(ckptpath), exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    best = float('inf')

    pbar = tqdm(range(args.epoch_main), desc="Training")
    for epoch in pbar:
        model = model.to(device)

        # training and validation using your project's helper functions
        train_loss = train_phase_one(train_loader, model, label_tokenizer, optimizer, device)
        val_loss = valid_phase_one(val_loader, model, label_tokenizer, device)

        # scheduler step (same as original)
        scheduler.step()

        # progress description (keeps original formatting)
        pbar.set_description(
            f"Phase 1 --Epoch {epoch + 1}/{args.epoch_main} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train: {train_loss:.4f}  | Val: {val_loss:.4f}"
        )
        # save best model by validation loss
        if val_loss < best:
            torch.save(model.state_dict(), ckptpath)
            best = val_loss

        # periodic test (every args.epoch_test)
        if epoch % args.epoch_test == 0:
            if os.path.exists(ckptpath):
                best_model = torch.load(ckptpath, map_location=device)
                model.load_state_dict(best_model)
                model = model.to(device)
                y_true, y_prob = test_phase_one(test_loader, model, label_tokenizer, device, show_progress=False)
                print(f"[Epoch {epoch}] Code Metrics: {code_level(y_true, y_prob)} | Visit Metrics: {visit_level(y_true, y_prob)}")
            else:
                # If checkpoint not yet saved, skip periodic test (mirrors original behavior if no ckpt)
                print(f"Test skipped at epoch {epoch}: no ckpt found yet.")

    # return path to checkpoint (experiments can decide further processing)
    return ckptpath

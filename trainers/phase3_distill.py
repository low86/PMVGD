# trainers/phase3_distill.py
"""
Phase 3 trainer: Adaptive sequential distillation.

This trainer performs the sequential distillation stage using the
auxiliary models to distill knowledge into the main model. It expects:
 - train_aux_loader, val_aux_loader, test_aux_loader prepared by experiments
 - main_model, aux_model1, aux_model2 instances already created and (for aux) loaded from phase2 ckpts
 Author: Tian Yuxuan
 Date: 2025-08-24
"""

import random
import argparse
import torch.nn.functional as F
from data.dataset import *
from models.encoder import *
from utils.dataloader import *
from data.subgraph_builder import *
from models.loss.contrast_loss import *
from utils.metapath_config import view_metas
from data.splits import split_ids_from_list_pkl
from torch.optim.lr_scheduler import CosineAnnealingLR


AUX_VIEW_NAMES = ("medication", "procedure")


@torch.no_grad()
def select_adaptive_view(train_aux_loader, main_model, aux_models, history_counts, history_penalty, device):
    """Select a student using mean cosine compatibility and selection history."""
    was_training = main_model.training
    main_model.eval()
    similarity_sums = [0.0 for _ in aux_models]
    patient_count = 0

    for seq_batch, graph_batch, drug_batch, procedure_batch, _ in train_aux_loader:
        _, patient_repr, teacher_repr = main_model((seq_batch, graph_batch))
        _, medication_repr = aux_models[0](drug_batch.to(device), patient_repr)
        _, procedure_repr = aux_models[1](procedure_batch.to(device), patient_repr)

        for index, student_repr in enumerate((medication_repr, procedure_repr)):
            similarity_sums[index] += F.cosine_similarity(
                teacher_repr, student_repr, dim=1
            ).sum().item()
        patient_count += teacher_repr.size(0)

    if was_training:
        main_model.train()

    similarities = [value / max(patient_count, 1) for value in similarity_sums]
    total_selections = sum(history_counts)
    normalized_history = [
        count / max(total_selections, 1) for count in history_counts
    ]
    selection_scores = [
        similarity - history_penalty * history
        for similarity, history in zip(similarities, normalized_history)
    ]
    selected_stage = max(range(len(aux_models)), key=selection_scores.__getitem__)
    return selected_stage, similarities, normalized_history, selection_scores

def run_phase3(train_aux_loader,
               val_aux_loader,
               test_aux_loader,
               main_model,
               aux_model1,
               aux_model2,
               label_tokenizer,
               args,
               device,
               ckpt_dir='./ckpt/phase3'):
    """
    Execute Phase 3 distillation.

    Args:
        train_aux_loader, val_aux_loader, test_aux_loader: DataLoaders prepared by experiments
        main_model: model to be trained (Teacher architecture reused)
        aux_model1, aux_model2: auxiliary models loaded from Phase2 pred ckpt
        label_tokenizer: label tokenizer
        args: argparse.Namespace with epoch_kd, lr, epoch_test, model, dataset, seed, etc.
        device: torch.device
        ckpt_dir: directory to save final model
    Returns:
        ckpt_path: path of saved phase3 checkpoint
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'phase3_{args.model}_{args.dataset}_{args.seed}.ckpt')

    optimizer = torch.optim.AdamW(list(main_model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    contrast = ContrastiveLearner(temperature=0.07)

    # hyperparameters used in your original script
    lambda_kd = 0.01
    lambda_repr = 0.03
    temperature = getattr(args, 'temperature', 2.0)

    aux_models = [aux_model1, aux_model2]
    aux_names = list(AUX_VIEW_NAMES)
    num_stages = len(aux_models)
    controller = getattr(args, 'distill_controller', 'normal')
    history_penalty = getattr(args, 'history_penalty', 0.03)
    switch_interval = getattr(args, 'switch_interval', 10)
    history_counts = [0 for _ in aux_models]

    best_code = float("-inf")
    best_visit = float("-inf")  # kept for parity with original (not used in save condition)

    pbar_kd = tqdm(range(args.epoch_kd), desc="Phase 3 Stage")

    for epoch in pbar_kd:
        if controller == 'adaptive':
            stage, _, _, _ = select_adaptive_view(
                train_aux_loader, main_model, aux_models, history_counts,
                history_penalty, device,
            )
        else:
            stage = (epoch // switch_interval) % num_stages

        history_counts[stage] += 1

        # train phase kd using the selected auxiliary view
        total_pred_loss, total_contrast_loss, total_kd_loss = train_phase_kd(
            aux_models, aux_names, stage, train_aux_loader, main_model,
            lambda_kd, lambda_repr, label_tokenizer, contrast, optimizer, device,
            temperature=temperature,
        )

        # evaluate on validation and compute val_loss
        metric_code, metric_visit = evaluate(val_aux_loader, main_model, label_tokenizer, device)
        val_loss = valid_phase_one(val_aux_loader, main_model, label_tokenizer, device)

        try:
            avg_contrast_loss = total_contrast_loss / len(train_aux_loader)
        except Exception:
            avg_contrast_loss = total_contrast_loss

        try:
            avg_pred_loss = total_pred_loss.item() / len(train_aux_loader)
        except Exception:
            avg_pred_loss = total_pred_loss / len(train_aux_loader)

        try:
            avg_kd_loss = total_kd_loss.item() / len(train_aux_loader)
        except Exception:
            avg_kd_loss = total_kd_loss / len(train_aux_loader)

        scheduler.step()

        pbar_kd.set_description(
            f"Phase 3 | {controller}:{aux_names[stage]} | Epoch {epoch + 1}/{args.epoch_kd} | "
            f"Repr: {avg_contrast_loss:.4f} | Pred: {avg_pred_loss:.4f} | "
            f"Kd: {avg_kd_loss:.4f} | Val_loss: {val_loss:.4f}"
        )

        if best_code < metric_code:
            best_code = metric_code
            torch.save(main_model.state_dict(), ckpt_path)

        if epoch % 10 == 0:
            if os.path.exists(ckpt_path):
                best_model = torch.load(ckpt_path, map_location=device)
                main_model.load_state_dict(best_model)
                model = main_model.to(device)
                y_true, y_prob = test_phase_one(test_aux_loader, model, label_tokenizer, device, show_progress=False)
                print(f"Test-{epoch // 10} Code Level Metrics: {code_level(y_true, y_prob)}")
                print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")
            else:
                print(f"Phase 3 periodic test skipped at epoch {epoch}: no ckpt yet.")

    # final: load best model and evaluate on test set
    if os.path.exists(ckpt_path):
        best_model = torch.load(ckpt_path, map_location=device)
        main_model.load_state_dict(best_model)
        model = main_model.to(device)
        y_true, y_prob = test_phase_one(test_aux_loader, model, label_tokenizer, device)
        print("Final Phase 3 evaluation (best ckpt):")
        print(f"Code Level Metrics: {code_level(y_true, y_prob)}")
        print(f"Visit Level Metrics: {visit_level(y_true, y_prob)}")
    else:
        print("Warning: phase3 checkpoint not found; nothing to evaluate.")

    return ckpt_path

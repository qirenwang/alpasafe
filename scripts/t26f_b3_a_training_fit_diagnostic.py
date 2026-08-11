#!/usr/bin/env python
"""Training-pool fit diagnostic for the six frozen final A/B checkpoints.

Purpose: separate the two explanations of the T26F-B3-A null result —
  (a) B overfits: B fits the 22-scene development pool much better than A;
  (b) B never uses L3: B ~= A on the training pool too, and zeroing /
      swapping L3 barely moves B's own training-pool behaviour.

READ-ONLY. Loads the six frozen checkpoints, the frozen B1 model class, the
frozen B1 loss functions, and the frozen full-development preprocessing, and
evaluates them on the 22 development scenes (their own training data). It
retrains nothing, touches no B3 artifact, and changes no frozen file. The
numbers here are training-fit diagnostics, NOT a held-out result.

For arm B it additionally evaluates the same checkpoint with
  - the correct L3 (NORMAL),
  - an all-zero L3 (ZERO),
  - a tag-matched different-scene L3 (WRONG_SCENE, deterministic cyclic
    shift over the 22 development scenes, never same-scene),
so the L3 sensitivity on TRAINING data can be compared against the
prospective B3 diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/qiren/alpasafe/safeworld-alpamayo")
B1 = REPO / ("artifacts/safeworld_t26f_b1_scenelatent_bounded_pilot/"
             "20260716T175100Z")
B30 = REPO / ("artifacts/safeworld_t26f_b3_0_preregistration/"
              "20260718T180730Z")
sys.path.insert(0, str(B1 / "code_artifacts"))
sys.path.insert(0, str(REPO / "src"))

import t26f_b1_train as b1  # noqa: E402  (frozen losses)
from t26f_b1_models import SafeWorldV2, parameter_count  # noqa: E402
from safeworld.t26e.train_pilot import K_VIEWS, group_metrics  # noqa: E402
from safeworld.t26e.scene_cv import rank0_regret_per_group  # noqa: E402

SEEDS = (0, 1, 2)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DevPool:
    """The 22-scene development pool with the frozen full-dev preprocessing."""

    def __init__(self, device):
        index = json.loads(
            (B1 / "datasets/t26f_b1_group_dataset_index.json").read_text())
        self.index = index["groups"]
        z = np.load(B1 / index["group_arrays_npz"])
        pre = json.loads(
            (B30 / "manifests/t26f_b3_0_full_dev_preprocessing_manifest.json")
            .read_text())
        mean = torch.tensor(pre["plan_mean_xy"], dtype=torch.float32,
                            device=device)
        std = torch.tensor(pre["plan_std_xy"], dtype=torch.float32,
                           device=device)
        aux = json.loads(
            (B30 / "manifests/t26f_b3_0_auxiliary_activation_manifest.json")
            .read_text())["fields"]
        self.aux_enabled = {f: aux[f]["enabled"] for f in aux}
        self.aux_pos_weight = {f: aux[f]["pos_weight_train_only"] for f in aux}

        from safetensors.torch import load_file
        self.groups = {}
        self.scene_of = {}
        self.tag_of = {}
        for gid, meta in sorted(self.index.items()):
            key = meta["npz_key"]
            latent = load_file(str(B1 / meta["latent_shard"]))
            raw = torch.from_numpy(z[f"planned__{key}"]).float().to(device)
            self.groups[gid] = {
                "tau_raw": raw, "tau_norm": (raw - mean) / std,
                "future": torch.from_numpy(
                    z[f"future__{key}"]).float().to(device),
                "future_tmask": torch.from_numpy(
                    z[f"future_tmask__{key}"]).to(device),
                "score": torch.from_numpy(
                    z[f"score__{key}"]).float().to(device),
                "progress": torch.from_numpy(
                    z[f"progress_clipped_rel__{key}"]).float().to(device),
                "collision": torch.from_numpy(
                    z[f"collision_at_fault__{key}"]).float().to(device),
                "offroad": torch.from_numpy(
                    z[f"offroad__{key}"]).float().to(device),
                "L3": latent["L3"].to(device),
            }
            self.scene_of[gid] = meta["scene_id"]
            self.tag_of[gid] = meta["decision_tag"]
        self.gids = sorted(self.groups)
        self.split_gids = {"train": self.gids}
        self.target = np.concatenate(
            [z[f"score__{self.index[g]['npz_key']}"] for g in self.gids])
        self.meta = [
            {"sample_id": f"{g}#cand_{i}", "decision_group_id": g,
             "candidate_id": f"cand_{i}", "candidate_index": i}
            for g in self.gids for i in range(8)]

    def latent_for(self, arm, group):
        return group["L3"] if arm == "B" else None

    def wrong_scene_map(self):
        """Deterministic tag-matched cyclic donor over the 22 dev scenes."""
        by_tag = {}
        for gid in self.gids:
            by_tag.setdefault(self.tag_of[gid], []).append(gid)
        mapping = {}
        for tag, gids in by_tag.items():
            ordered = sorted(gids, key=lambda g: self.scene_of[g])
            for i, gid in enumerate(ordered):
                mapping[gid] = ordered[(i + 1) % len(ordered)]
        return mapping


@torch.no_grad()
def evaluate(model, pool, arm, latent_mode, wrong_map):
    """Per-component losses + selector metrics on the development pool."""
    model.eval()
    parts = {"score": [], "future": [], "progress": [],
             "collision_at_fault": [], "offroad": []}
    pair_losses = []
    predictions = np.zeros(len(pool.gids) * 8, dtype=np.float32)
    for j, gid in enumerate(pool.gids):
        group = pool.groups[gid]
        if arm == "A":
            latent = None
        elif latent_mode == "NORMAL":
            latent = group["L3"]
        elif latent_mode == "ZERO":
            latent = torch.zeros_like(group["L3"])
        else:
            latent = pool.groups[wrong_map[gid]]["L3"]
        out = model(group["tau_norm"], group["tau_raw"], latent)
        predictions[j * 8:(j + 1) * 8] = (
            out["predicted_score"].cpu().numpy().astype(np.float32))
        import torch.nn.functional as F
        parts["score"].append(
            F.smooth_l1_loss(out["predicted_score"], group["score"], beta=1.0))
        ii, jj = torch.triu_indices(8, 8, offset=1)
        delta = group["score"][ii] - group["score"][jj]
        keep = delta.abs() > b1.RANK_TIE_EPS
        if keep.any():
            sign = torch.sign(delta[keep])
            logits = out["score_logit"]
            pair_losses.append(F.softplus(
                -sign * (logits[ii][keep] - logits[jj][keep])))
        fmask = group["future_tmask"].unsqueeze(-1).expand_as(group["future"])
        parts["future"].append(F.smooth_l1_loss(
            out["predicted_future"][fmask], group["future"][fmask], beta=1.0))
        parts["progress"].append(F.smooth_l1_loss(
            out["progress_pred"], group["progress"], beta=1.0))
        for field, logit_key in (("collision_at_fault", "collision_logit"),
                                 ("offroad", "offroad_logit")):
            if pool.aux_enabled[field]:
                weight = torch.tensor(pool.aux_pos_weight[field],
                                      device=group["score"].device)
                target = (group["collision"] if field == "collision_at_fault"
                          else group["offroad"])
                parts[field].append(F.binary_cross_entropy_with_logits(
                    out[logit_key], target, pos_weight=weight))
    losses = {k: (float(torch.stack(v).mean()) if v else None)
              for k, v in parts.items()}
    losses["rank"] = float(torch.cat(pair_losses).mean()) if pair_losses else None
    total = losses["score"] + losses["rank"] + losses["future"] \
        + 0.25 * losses["progress"]
    for field in ("collision_at_fault", "offroad"):
        if losses[field] is not None:
            total += 0.25 * losses[field]
    losses["total"] = total

    metrics = {}
    for name, k in K_VIEWS.items():
        gm = group_metrics(predictions, pool.target, pool.meta, k)
        r0 = rank0_regret_per_group(pool.target, pool.meta, k)
        metrics[name] = {
            "mean_selected_regret": gm["mean_selected_score_regret"],
            "top1_accuracy": gm["top1_selection_accuracy"],
            "pairwise_accuracy": gm["pairwise_ranking_accuracy"],
            "rank0_mean_regret": float(np.mean(list(r0.values()))),
        }
    return {"losses": losses, "selector_metrics": metrics,
            "prediction_std": float(predictions.std()),
            "prediction_mean": float(predictions.mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = DevPool(device)
    wrong_map = pool.wrong_scene_map()
    assert all(pool.scene_of[k] != pool.scene_of[v]
               for k, v in wrong_map.items())
    assert all(pool.tag_of[k] == pool.tag_of[v] for k, v in wrong_map.items())

    ckpt_manifest = json.loads(
        (B30 / "manifests/t26f_b3_0_final_checkpoint_manifest.json")
        .read_text())["sha256"]

    results = {}
    for arm in ("A", "B"):
        for seed in SEEDS:
            rel = f"models/final_ab/final_{arm}_seed{seed}.pt"
            path = B30 / rel
            assert sha_file(path) == ckpt_manifest[rel], f"hash drift: {rel}"
            model = SafeWorldV2(arm).to(device)
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state["model_state"])
            modes = ["NORMAL"] if arm == "A" else ["NORMAL", "ZERO",
                                                   "WRONG_SCENE"]
            for mode in modes:
                key = f"{arm}_seed{seed}" + ("" if mode == "NORMAL"
                                             else f"_{mode}")
                results[key] = evaluate(model, pool, arm, mode, wrong_map)
                results[key]["fixed_epochs_trained"] = state[
                    "fixed_epochs_trained"]
                results[key]["parameters"] = parameter_count(model)["total"]
            del model

    def mean_over_seeds(prefix, field_path):
        values = []
        for seed in SEEDS:
            node = results[f"{prefix}{seed}" if prefix.endswith("seed")
                           else prefix.replace("SEED", str(seed))]
            for key in field_path:
                node = node[key]
            values.append(node)
        return float(np.mean(values))

    summary = {
        "A_train_total_loss": mean_over_seeds("A_seedSEED".replace("SEED", "SEED"), []) if False else float(np.mean([results[f"A_seed{s}"]["losses"]["total"] for s in SEEDS])),
        "B_train_total_loss": float(np.mean([results[f"B_seed{s}"]["losses"]["total"] for s in SEEDS])),
        "A_train_L_score": float(np.mean([results[f"A_seed{s}"]["losses"]["score"] for s in SEEDS])),
        "B_train_L_score": float(np.mean([results[f"B_seed{s}"]["losses"]["score"] for s in SEEDS])),
        "A_train_L_rank": float(np.mean([results[f"A_seed{s}"]["losses"]["rank"] for s in SEEDS])),
        "B_train_L_rank": float(np.mean([results[f"B_seed{s}"]["losses"]["rank"] for s in SEEDS])),
        "A_train_K8_regret": float(np.mean([results[f"A_seed{s}"]["selector_metrics"]["K8"]["mean_selected_regret"] for s in SEEDS])),
        "B_train_K8_regret": float(np.mean([results[f"B_seed{s}"]["selector_metrics"]["K8"]["mean_selected_regret"] for s in SEEDS])),
        "B_train_K8_regret_ZERO_L3": float(np.mean([results[f"B_seed{s}_ZERO"]["selector_metrics"]["K8"]["mean_selected_regret"] for s in SEEDS])),
        "B_train_K8_regret_WRONG_L3": float(np.mean([results[f"B_seed{s}_WRONG_SCENE"]["selector_metrics"]["K8"]["mean_selected_regret"] for s in SEEDS])),
        "rank0_train_K8_regret": results["A_seed0"]["selector_metrics"]["K8"]["rank0_mean_regret"],
    }
    summary["B_minus_A_train_K8_regret"] = (
        summary["B_train_K8_regret"] - summary["A_train_K8_regret"])
    summary["B_zero_minus_normal_train_K8"] = (
        summary["B_train_K8_regret_ZERO_L3"] - summary["B_train_K8_regret"])
    summary["B_wrong_minus_normal_train_K8"] = (
        summary["B_train_K8_regret_WRONG_L3"] - summary["B_train_K8_regret"])

    payload = {
        "task": "t26f_b3_a_training_fit_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "separate overfitting from non-use of L3 by measuring the "
                   "six frozen final checkpoints on their OWN training data "
                   "(the 22-scene development pool)",
        "read_only": True,
        "retrained_anything": False,
        "b3_artifacts_touched": False,
        "caveat": "these are TRAINING-FIT numbers on data the models were "
                  "fitted on; they are not held-out results and must never "
                  "be reported as performance evidence",
        "wrong_scene_donor_rule": "tag-matched cyclic shift over the 22 "
                                  "development scenes, never same-scene",
        "summary": summary,
        "per_model": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

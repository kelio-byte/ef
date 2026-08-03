from rdkit import Chem
import os
import argparse
from functools import partial
import hashlib
import json
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import re

from preprocessing.global_align import inverse_global_align

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smi_tokenizer(smi):
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def validate_scoring_options(opt):
    """Validate shape-related CLI options before reading large files."""
    if opt.augmentation <= 0:
        raise ValueError(
            f"augmentation must be > 0, got {opt.augmentation}"
        )
    if opt.beam_size <= 0:
        raise ValueError(f"beam_size must be > 0, got {opt.beam_size}")
    if opt.n_best <= 0:
        raise ValueError(f"n_best must be > 0, got {opt.n_best}")
    if opt.process_number <= 0:
        raise ValueError(
            f"process_number must be > 0, got {opt.process_number}"
        )
    if opt.length == 0 or opt.length < -1:
        raise ValueError(f"length must be -1 or > 0, got {opt.length}")
    if getattr(opt, "target_offset", 0) < 0:
        raise ValueError(
            f"target_offset must be >= 0, got {opt.target_offset}"
        )
    if opt.raw and opt.augmentation != 1:
        raise ValueError(
            "raw scoring requires augmentation=1, got "
            f"{opt.augmentation}"
        )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_metadata(
    metadata,
    metadata_path,
    prediction_path,
    prediction_count,
    augmentation,
    beam_size,
    target_offset=0,
):
    """Cross-check a sampling or mixing manifest against scorer inputs."""
    required = ("output_beam_size", "output_line_count", "output_sha256")
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(
            f"prediction metadata {metadata_path} is missing required "
            f"fields: {', '.join(missing)}"
        )

    metadata_beam_size = metadata["output_beam_size"]
    if metadata_beam_size != beam_size:
        raise ValueError(
            "beam_size does not match prediction metadata: metadata has "
            f"{metadata_beam_size}, scorer received {beam_size}."
        )
    metadata_augmentation = metadata.get("augmentation")
    if (metadata_augmentation is not None
            and metadata_augmentation != augmentation):
        raise ValueError(
            "augmentation does not match prediction metadata: metadata has "
            f"{metadata_augmentation}, scorer received {augmentation}."
        )
    if metadata["output_line_count"] != prediction_count:
        raise ValueError(
            "prediction line count does not match metadata: metadata has "
            f"{metadata['output_line_count']}, file has {prediction_count}."
        )

    actual_sha256 = _sha256_file(prediction_path)
    if metadata["output_sha256"] != actual_sha256:
        raise ValueError(
            "prediction SHA-256 does not match metadata; the predictions "
            "file or metadata was changed after generation."
        )

    if "product_count" in metadata:
        expected_count = metadata["product_count"] * metadata_beam_size
        if expected_count != prediction_count:
            raise ValueError(
                "sampling metadata product_count is inconsistent with its "
                f"output layout: expected {expected_count} lines, got "
                f"{prediction_count}."
            )
    if "reaction_count" in metadata:
        expected_count = (
            metadata["reaction_count"]
            * augmentation
            * metadata_beam_size
        )
        if expected_count != prediction_count:
            raise ValueError(
                "mixing metadata reaction_count is inconsistent with its "
                f"output layout: expected {expected_count} lines, got "
                f"{prediction_count}."
            )

    input_metadata = metadata.get("input", {})
    selection_start = input_metadata.get("selection_start_product")
    if selection_start is not None and metadata_augmentation is not None:
        expected_start = target_offset * augmentation
        if selection_start != expected_start:
            raise ValueError(
                "target_offset does not match sampling metadata: selected "
                f"product line starts at {selection_start}, but "
                f"target_offset={target_offset} and augmentation="
                f"{augmentation} imply {expected_start}."
            )


def load_and_validate_prediction_metadata(
    prediction_path,
    prediction_count,
    augmentation,
    beam_size,
    target_offset=0,
):
    parent = os.path.dirname(os.path.abspath(prediction_path))
    candidates = [
        os.path.join(parent, filename)
        for filename in ("sampling_metadata.json", "mixing_metadata.json")
        if os.path.exists(os.path.join(parent, filename))
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "multiple prediction metadata files found; remove the stale "
            f"manifest before scoring: {candidates}"
        )

    metadata_path = candidates[0]
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    validate_prediction_metadata(
        metadata,
        metadata_path=metadata_path,
        prediction_path=prediction_path,
        prediction_count=prediction_count,
        augmentation=augmentation,
        beam_size=beam_size,
        target_offset=target_offset,
    )
    return metadata_path


def resolve_input_layout(
    prediction_count,
    target_count,
    augmentation,
    beam_size,
    length=-1,
):
    """Return the reaction count and consumed line counts.

    Without ``--length``, prediction and target files must describe exactly the
    same number of reactions.  With ``--length``, a prefix is intentional, but
    both files must contain enough complete rows for that prefix.
    """
    prediction_block = augmentation * beam_size
    if prediction_count <= 0:
        raise ValueError("prediction file is empty")

    if length == -1:
        if prediction_count % prediction_block != 0:
            remainder = prediction_count % prediction_block
            raise ValueError(
                "prediction line count is not divisible by "
                "augmentation * beam_size: "
                f"{prediction_count} % ({augmentation} * {beam_size}) "
                f"= {remainder}. Check the prediction file and beam_size."
            )
        data_size = prediction_count // prediction_block
        expected_targets = data_size * augmentation
        if target_count != expected_targets:
            raise ValueError(
                "target line count does not match predictions: expected "
                f"{expected_targets} lines for {data_size} reactions, got "
                f"{target_count}."
            )
        return data_size, prediction_count, expected_targets

    data_size = length
    required_predictions = data_size * prediction_block
    required_targets = data_size * augmentation
    if prediction_count < required_predictions:
        raise ValueError(
            f"--length {length} requires {required_predictions} prediction "
            f"lines, got {prediction_count}."
        )
    if target_count < required_targets:
        raise ValueError(
            f"--length {length} requires {required_targets} target lines, "
            f"got {target_count}."
        )
    return data_size, required_predictions, required_targets


def canonicalize_smiles_clear_map(
    smiles,
    return_max_frag=True,
    synthon=False,
):
    smiles = inverse_global_align(smiles)
    mol = Chem.MolFromSmiles(smiles,sanitize=not synthon)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '',''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles,sanitize=not synthon) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(
                    sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],
                    return_max_frag=False,
                    synthon=synthon,
                )
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''


def _deduplicate_valid(candidates):
    """Remove invalid and repeated canonical candidates, preserving order."""
    deduplicated = []
    seen = set()
    for candidate in candidates:
        if candidate[0] == "" or candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def compute_rank(
    prediction,
    raw=False,
    alpha=1.0,
    beam_size=None,
    aggregation_mode="legacy_best_rank",
):
    if not prediction or not prediction[0]:
        raise ValueError("prediction must contain at least one candidate")
    if beam_size is None:
        beam_size = len(prediction[0])
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    highest = {}
    support = {}
    if raw:
        # no test augmentation
        assert len(prediction) == 1
        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                if prediction[j][k][0] == "":
                    invalid_rates[k] += 1
            # error detection
            valid_prediction = [i for i in prediction[j] if i[0] != ""]
            for k, data in enumerate(valid_prediction):
                rank[data] = 1 / (alpha * k + 1)
    else:

        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
                if prediction[j][k][0] == "":
                    valid_score[j][k] = beam_size + 1
                    invalid_rates[k] += 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0][0] != ""]
            unique_prediction = _deduplicate_valid(de_error)
            for k, data in enumerate(unique_prediction):
                if data in rank:
                    rank[data] += 1 / (alpha * k + 1)
                else:
                    rank[data] = 1 / (alpha * k + 1)
                if data in highest:
                    highest[data] = min(k,highest[data])
                else:
                    highest[data] = k
                support[data] = support.get(data, 0) + 1
        for key in rank.keys():
            if aggregation_mode == "legacy_best_rank":
                rank[key] += highest[key] * -1e8
            elif aggregation_mode == "rrf":
                pass
            elif aggregation_mode == "frequency_first":
                rank[key] += support[key] * 1e8
            elif aggregation_mode == "hybrid":
                rank[key] += 1.0 / (highest[key] + 1)
            else:
                raise ValueError(
                    f"unsupported aggregation_mode: {aggregation_mode}"
                )
    return rank,invalid_rates


def compute_sampling_diagnostics(
    predictions,
    ground_truth,
    beam_size,
    alpha=1.0,
    top_k=3,
    report_n_best=None,
    raw=False,
    aggregation_mode="legacy_best_rank",
):
    """Measure sampling coverage separately from aggregation quality."""
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "predictions and ground_truth must contain the same number of "
            f"reactions, got {len(predictions)} and {len(ground_truth)}"
        )

    data_size = len(predictions)
    invalid_by_run = [0] * beam_size
    valid_by_run = [0] * beam_size
    duplicate_by_run = [0] * beam_size
    target_hit_by_run = [0] * beam_size
    overlap_intersections = {}
    overlap_unions = {}
    overlap_macro_sum = {}
    overlap_macro_count = {}
    for left in range(beam_size):
        for right in range(left + 1, beam_size):
            key = f"run_{left + 1}_vs_{right + 1}"
            overlap_intersections[key] = 0
            overlap_unions[key] = 0
            overlap_macro_sum[key] = 0.0
            overlap_macro_count[key] = 0

    oracle_hits = 0
    covered_outside_top_k = 0
    target_aug_counts = []
    covered_final_ranks = []
    best_local_rank_counts = [0] * beam_size
    true_unique_counts = []
    valid_candidate_counts = []
    per_reaction = []

    for reaction_index, (prediction, target) in enumerate(
        zip(predictions, ground_truth)
    ):
        target_smiles = target[0]
        run_sets = []
        for run_index in range(beam_size):
            run_candidates = [
                prediction[aug_index][run_index][0]
                for aug_index in range(len(prediction))
            ]
            valid_candidates = [smi for smi in run_candidates if smi != ""]
            invalid_by_run[run_index] += (
                len(run_candidates) - len(valid_candidates)
            )
            valid_by_run[run_index] += len(valid_candidates)
            unique_candidates = set(valid_candidates)
            duplicate_by_run[run_index] += (
                len(valid_candidates) - len(unique_candidates)
            )
            target_hit = target_smiles != "" and target_smiles in unique_candidates
            target_hit_by_run[run_index] += int(target_hit)
            run_sets.append(unique_candidates)

        for left in range(beam_size):
            for right in range(left + 1, beam_size):
                key = f"run_{left + 1}_vs_{right + 1}"
                intersection = run_sets[left] & run_sets[right]
                union = run_sets[left] | run_sets[right]
                overlap_intersections[key] += len(intersection)
                overlap_unions[key] += len(union)
                if union:
                    overlap_macro_sum[key] += len(intersection) / len(union)
                    overlap_macro_count[key] += 1

        all_valid = set().union(*run_sets)
        valid_count = sum(
            1
            for augmentation in prediction
            for candidate in augmentation
            if candidate[0] != ""
        )
        true_unique_counts.append(len(all_valid))
        valid_candidate_counts.append(valid_count)

        target_aug_count = 0
        best_local_rank = None
        for augmentation in prediction:
            local_candidates = _deduplicate_valid(augmentation)
            local_full_smiles = [candidate[0] for candidate in local_candidates]
            if target_smiles != "" and target_smiles in local_full_smiles:
                target_aug_count += 1
                local_rank = local_full_smiles.index(target_smiles)
                if best_local_rank is None or local_rank < best_local_rank:
                    best_local_rank = local_rank

        rank_scores, _ = compute_rank(
            prediction,
            raw=raw,
            alpha=alpha,
            beam_size=beam_size,
            aggregation_mode=aggregation_mode,
        )
        ranked_candidates = sorted(
            rank_scores.items(), key=lambda item: item[1], reverse=True
        )
        final_rank = None
        consensus_score = None
        for rank_index, (candidate, score) in enumerate(ranked_candidates):
            if candidate[0] == target_smiles:
                final_rank = rank_index
                consensus_score = score
                break

        oracle_any = target_smiles != "" and target_smiles in all_valid
        oracle_hits += int(oracle_any)
        target_aug_counts.append(target_aug_count)
        if best_local_rank is not None and best_local_rank < beam_size:
            best_local_rank_counts[best_local_rank] += 1
        if final_rank is not None:
            covered_final_ranks.append(final_rank)
        if oracle_any and (final_rank is None or final_rank >= top_k):
            covered_outside_top_k += 1

        per_reaction.append({
            "reaction_index": reaction_index,
            "oracle_any": oracle_any,
            "target_augmentation_count": target_aug_count,
            "target_best_local_rank": (
                None if best_local_rank is None else best_local_rank + 1
            ),
            "target_consensus_score": consensus_score,
            "target_final_rank": None if final_rank is None else final_rank + 1,
            "true_unique_candidates": len(all_valid),
            "valid_candidate_count": valid_count,
        })

    candidate_slots_per_run = (
        data_size * len(predictions[0]) if data_size else 0
    )
    run_metrics = []
    for run_index in range(beam_size):
        valid_count = valid_by_run[run_index]
        run_metrics.append({
            "run": run_index + 1,
            "target_hit_count": target_hit_by_run[run_index],
            "target_hit_rate_percent": (
                100.0 * target_hit_by_run[run_index] / data_size
                if data_size else 0.0
            ),
            "invalid_count": invalid_by_run[run_index],
            "invalid_rate_percent": (
                100.0 * invalid_by_run[run_index] / candidate_slots_per_run
                if candidate_slots_per_run else 0.0
            ),
            "duplicate_count": duplicate_by_run[run_index],
            "duplicate_rate_among_valid_percent": (
                100.0 * duplicate_by_run[run_index] / valid_count
                if valid_count else 0.0
            ),
        })

    pairwise_overlap = []
    for key in overlap_intersections:
        union_count = overlap_unions[key]
        macro_count = overlap_macro_count[key]
        pairwise_overlap.append({
            "pair": key,
            "micro_jaccard_percent": (
                100.0 * overlap_intersections[key] / union_count
                if union_count else 0.0
            ),
            "mean_reaction_jaccard_percent": (
                100.0 * overlap_macro_sum[key] / macro_count
                if macro_count else 0.0
            ),
        })

    if report_n_best is None:
        report_n_best = top_k
    aggregated_rank_availability = {
        str(rank): (
            100.0 * sum(count >= rank for count in true_unique_counts)
            / data_size
            if data_size else 0.0
        )
        for rank in range(1, report_n_best + 1)
    }

    summary = {
        "reaction_count": data_size,
        "oracle_any_count": oracle_hits,
        "oracle_any_percent": (
            100.0 * oracle_hits / data_size if data_size else 0.0
        ),
        "covered_outside_top_k": covered_outside_top_k,
        "covered_outside_top_k_percent": (
            100.0 * covered_outside_top_k / data_size
            if data_size else 0.0
        ),
        "top_k_for_coverage_loss": top_k,
        "mean_target_augmentation_count": (
            sum(target_aug_counts) / data_size if data_size else 0.0
        ),
        "mean_target_final_rank_when_covered": (
            sum(rank + 1 for rank in covered_final_ranks)
            / len(covered_final_ranks)
            if covered_final_ranks else None
        ),
        "best_local_rank_counts": {
            str(rank + 1): count
            for rank, count in enumerate(best_local_rank_counts)
        },
        "mean_valid_candidates_per_reaction": (
            sum(valid_candidate_counts) / data_size if data_size else 0.0
        ),
        "mean_true_unique_candidates_per_reaction": (
            sum(true_unique_counts) / data_size if data_size else 0.0
        ),
        "aggregated_rank_availability_percent": (
            aggregated_rank_availability
        ),
        "run_metrics": run_metrics,
        "pairwise_overlap": pairwise_overlap,
    }
    return {"summary": summary, "per_reaction": per_reaction}


def print_sampling_diagnostics(diagnostics):
    summary = diagnostics["summary"]
    top_k = summary["top_k_for_coverage_loss"]
    print("\nSampling coverage diagnostics:")
    print(
        "Oracle-any: "
        f"{summary['oracle_any_percent']:.3f}% "
        f"({summary['oracle_any_count']}/{summary['reaction_count']})"
    )
    print(
        f"Covered but outside Top-{top_k}: "
        f"{summary['covered_outside_top_k_percent']:.3f}% "
        f"({summary['covered_outside_top_k']}/"
        f"{summary['reaction_count']})"
    )
    print(
        "Mean target augmentation support: "
        f"{summary['mean_target_augmentation_count']:.3f}"
    )
    mean_final_rank = summary["mean_target_final_rank_when_covered"]
    if mean_final_rank is not None:
        print(f"Mean target final rank when covered: {mean_final_rank:.3f}")
    local_rank_text = ", ".join(
        f"rank {rank}: {count}"
        for rank, count in summary["best_local_rank_counts"].items()
    )
    print(f"Best local target rank counts: {local_rank_text}")
    print(
        "Mean valid / true unique candidates per reaction: "
        f"{summary['mean_valid_candidates_per_reaction']:.3f} / "
        f"{summary['mean_true_unique_candidates_per_reaction']:.3f}"
    )
    rank_availability_text = ", ".join(
        f"rank {rank}: {percent:.3f}%"
        for rank, percent in
        summary["aggregated_rank_availability_percent"].items()
    )
    print(f"Aggregated candidate availability: {rank_availability_text}")
    for metric in summary["run_metrics"]:
        print(
            f"Input rank {metric['run']}: target-hit "
            f"{metric['target_hit_rate_percent']:.3f}%, invalid "
            f"{metric['invalid_rate_percent']:.3f}%, duplicate-among-valid "
            f"{metric['duplicate_rate_among_valid_percent']:.3f}%"
        )
    for overlap in summary["pairwise_overlap"]:
        rank_pair = overlap["pair"].replace("run_", "rank_")
        print(
            f"Overlap input {rank_pair}: micro/macro Jaccard "
            f"{overlap['micro_jaccard_percent']:.3f}% / "
            f"{overlap['mean_reaction_jaccard_percent']:.3f}%"
        )


def main(opt):
    validate_scoring_options(opt)
    print('Reading predictions from file ...')
    print(f"Aggregation mode: {opt.aggregation_mode}")
    with open(opt.predictions, 'r') as f:
        prediction_lines = f.readlines()
    with open(opt.targets, 'r') as f:
        target_lines = f.readlines()

    print("Prediction File Length", len(prediction_lines))
    print("Origin Target File Length", len(target_lines))
    metadata_path = load_and_validate_prediction_metadata(
        prediction_path=opt.predictions,
        prediction_count=len(prediction_lines),
        augmentation=opt.augmentation,
        beam_size=opt.beam_size,
        target_offset=opt.target_offset,
    )
    if metadata_path:
        print(f"Validated prediction metadata: {metadata_path}")
    else:
        print("Prediction metadata: not found (legacy text-only input)")
    target_start_line = opt.target_offset * opt.augmentation
    if target_start_line >= len(target_lines):
        raise ValueError(
            "target_offset starts beyond the target file: line "
            f"{target_start_line} for {len(target_lines)} lines"
        )
    if target_start_line:
        target_lines = target_lines[target_start_line:]
        print(
            f"Applied target offset: reaction {opt.target_offset} "
            f"(line {target_start_line})"
        )
    data_size, used_prediction_count, used_target_count = resolve_input_layout(
        prediction_count=len(prediction_lines),
        target_count=len(target_lines),
        augmentation=opt.augmentation,
        beam_size=opt.beam_size,
        length=opt.length,
    )
    print(
        "Validated layout: "
        f"{data_size} reactions x {opt.augmentation} augmentations x "
        f"{opt.beam_size} candidates = {used_prediction_count} "
        "prediction lines"
    )
    if opt.length != -1 and (
        len(prediction_lines) > used_prediction_count
        or len(target_lines) > used_target_count
    ):
        print(
            f"Scoring the first {data_size} reactions because "
            f"--length={opt.length}."
        )

    prediction_lines = [
        ''.join(line.strip().split(' '))
        for line in prediction_lines[:used_prediction_count]
    ]
    canonicalizer = partial(
        canonicalize_smiles_clear_map,
        synthon=opt.synthon,
    )
    print("Canonicalizing predictions using Process Number ",opt.process_number)
    with multiprocessing.Pool(processes=opt.process_number) as pool:
        raw_predictions = list(tqdm(
            pool.imap(func=canonicalizer, iterable=prediction_lines),
            total=len(prediction_lines),
        ))

    predictions = [
        [[] for _ in range(opt.augmentation)]
        for _ in range(data_size)
    ]  # data_len x augmentation x beam_size
    prediction_block = opt.beam_size * opt.augmentation
    for i, line in enumerate(raw_predictions):
        reaction_index = i // prediction_block
        augmentation_index = (i % prediction_block) // opt.beam_size
        predictions[reaction_index][augmentation_index].append(line)

    print("data size ",data_size)
    print('Reading targets from file ...')
    target_lines = target_lines[:used_target_count]
    targets = [
        ''.join(target_lines[i].strip().split(' '))
        for i in range(0, used_target_count, opt.augmentation)
    ]
    with multiprocessing.Pool(processes=opt.process_number) as pool:
        targets = list(tqdm(
            pool.imap(func=canonicalizer, iterable=targets),
            total=len(targets),
        ))
    ground_truth = targets
    print("Canonical Target Length", len(ground_truth))
    accuracy = [0 for j in range(opt.n_best)]
    topn_accuracy_chirality = [0 for _ in range(opt.n_best)]
    topn_accuracy_wochirality = [0 for _ in range(opt.n_best)]
    topn_accuracy_ringopening = [0 for _ in range(opt.n_best)]
    topn_accuracy_ringformation = [0 for _ in range(opt.n_best)]
    topn_accuracy_woring = [0 for _ in range(opt.n_best)]
    total_chirality = 0
    total_ringopening = 0
    total_ringformation = 0
    atomsize_topk = []
    accurate_indices = [[] for j in range(opt.n_best)]
    max_frag_accuracy = [0 for j in range(opt.n_best)]
    invalid_rates = [0 for j in range(opt.beam_size)]
    sorted_invalid_rates = [0 for j in range(opt.beam_size)]
    unique_rates = 0
    ranked_results = []
    if opt.detailed:
        if not os.path.exists(opt.sources):
            print("Detailed Mode needs the sources.")
            exit(1)
        with open(opt.sources,"r") as f:
            lines = f.readlines()
            source_start = opt.target_offset * opt.augmentation
            required_sources = data_size * opt.augmentation
            if len(lines) < source_start + required_sources:
                raise ValueError(
                    "source file does not contain enough lines for "
                    f"target_offset={opt.target_offset} and {data_size} "
                    "reactions"
                )
            lines = lines[source_start:source_start + required_sources]
            ras_src_smiles = [''.join(lines[i].strip().split(' ')) for i in tqdm(range(0,data_size * opt.augmentation,opt.augmentation))]

    for i in tqdm(range(len(predictions))):
        accurate_flag = False
        if opt.detailed:
            chirality_flag = False
            ringopening_flag = False
            ringformation_flag = False
            pro_mol = Chem.MolFromSmiles(ras_src_smiles[i])
            rea_mol = Chem.MolFromSmiles(ground_truth[i][0])
            pro_ringinfo = pro_mol.GetRingInfo()
            rea_ringinfo = rea_mol.GetRingInfo()
            pro_ringnum = pro_ringinfo.NumRings()
            rea_ringnum = rea_ringinfo.NumRings()
            size = len(rea_mol.GetAtoms()) - len(pro_mol.GetAtoms())
            # if (int(ras_src_smiles[i].count("@") > 0) + int(ground_truth[i][0].count("@") > 0)) == 1:
            if ras_src_smiles[i].count("@") > 0 or ground_truth[i][0].count("@") > 0:
                total_chirality += 1
                chirality_flag = True
            if pro_ringnum < rea_ringnum:
                total_ringopening += 1
                ringopening_flag = True
            if pro_ringnum > rea_ringnum:
                total_ringformation += 1
                ringformation_flag = True

        rank, invalid_rate = compute_rank(
            predictions[i],
            raw=opt.raw,
            alpha=opt.score_alpha,
            beam_size=opt.beam_size,
            aggregation_mode=opt.aggregation_mode,
        )
        for j in range(opt.beam_size):
            invalid_rates[j] += invalid_rate[j]
        rank = list(zip(rank.keys(),rank.values()))
        rank.sort(key=lambda x:x[1],reverse=True)
        rank = rank[:opt.n_best]
        ranked_results.append([item[0][0] for item in rank])

        for j, item in enumerate(rank):
            if item[0][0] == ground_truth[i][0]:
                if not accurate_flag:
                    accurate_flag = True
                    accurate_indices[j].append(i)
                    for k in range(j, opt.n_best):
                        accuracy[k] += 1
                    if opt.detailed:
                        atomsize_topk.append((size,j))
                        if chirality_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_chirality[k] += 1
                        else:
                            for k in range(j,opt.n_best):
                                topn_accuracy_wochirality[k] += 1
                        if ringopening_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_ringopening[k] += 1
                        if ringformation_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_ringformation[k] += 1
                        if not ringopening_flag and not ringformation_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_woring[k] += 1

        if opt.detailed and not accurate_flag:
            atomsize_topk.append((size,opt.n_best))
        for j, item in enumerate(rank):
            if item[0][1] == ground_truth[i][1]:
                for k in range(j,opt.n_best):
                    max_frag_accuracy[k] += 1
                break
        for j in range(len(rank),opt.beam_size):
            sorted_invalid_rates[j] += 1
        unique_rates += len(rank)

    reported_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 49]
    for i in range(opt.n_best):
        if i in reported_ranks and i < opt.beam_size:
            print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i+1,accuracy[i] / data_size * 100,max_frag_accuracy[i] / data_size * 100),
                  " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(invalid_rates[i] / data_size / opt.augmentation * 100,sorted_invalid_rates[i] / data_size / opt.augmentation * 100))
        elif i in reported_ranks:
            print(
                "Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%, "
                "Invalid SMILES:N/A (no corresponding input rank)".format(
                    i + 1,
                    accuracy[i] / data_size * 100,
                    max_frag_accuracy[i] / data_size * 100,
                )
            )

    legacy_unique_rate = unique_rates / data_size / opt.beam_size * 100
    print("Unique Rates:{:.3f}%".format(legacy_unique_rate))
    print(
        "  Note: Unique Rates is a legacy metric based on rank[:n_best], "
        "not raw sampling diversity."
    )

    if opt.diagnostics or opt.diagnostics_json:
        diagnostics = compute_sampling_diagnostics(
            predictions,
            ground_truth,
            beam_size=opt.beam_size,
            alpha=opt.score_alpha,
            top_k=min(3, opt.n_best),
            report_n_best=opt.n_best,
            raw=opt.raw,
            aggregation_mode=opt.aggregation_mode,
        )
        if opt.diagnostics:
            print_sampling_diagnostics(diagnostics)
        if opt.diagnostics_json:
            output_parent = os.path.dirname(opt.diagnostics_json)
            if output_parent:
                os.makedirs(output_parent, exist_ok=True)
            with open(opt.diagnostics_json, "w") as f:
                json.dump(diagnostics, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"Diagnostics saved to: {opt.diagnostics_json}")

    if opt.detailed:
        print_topk = [1,3,5,10]
        save_dict = {}
        atomsize_topk.sort(key=lambda x:x[0])
        differ_now = atomsize_topk[0][0]
        topn_accuracy_bydiffer = [0 for _ in range(opt.n_best)]
        total_bydiffer = 0
        for i,item in enumerate(atomsize_topk):
            if differ_now < 11 and differ_now != item[0]:
                for j in range(opt.n_best):
                    if (j+1) in print_topk:
                        save_dict[f'top-{j+1}_size_{differ_now}'] = topn_accuracy_bydiffer[j] / total_bydiffer * 100
                        print("Top-{} Atom differ size {} Acc:{:.3f}%, Number:{:.3f}%".format(j+1,
                                              differ_now,
                                               topn_accuracy_bydiffer[j] / total_bydiffer * 100,
                                               total_bydiffer/data_size * 100))
                total_bydiffer = 0
                topn_accuracy_bydiffer = [0 for _ in range(opt.n_best)]
                differ_now = item[0]
            for k in range(item[1],opt.n_best):
                topn_accuracy_bydiffer[k] += 1
            total_bydiffer += 1
        for j in range(opt.n_best):
            if (j + 1) in print_topk:
                print("Top-{} Atom differ size {} Acc:{:.3f}%, Number:{:.3f}%".format(j + 1,
                      differ_now,
                      topn_accuracy_bydiffer[j] / total_bydiffer * 100,
                      total_bydiffer / data_size * 100))
                save_dict[f'top-{j+1}_size_{differ_now}'] = topn_accuracy_bydiffer[j] / total_bydiffer * 100

        for i in range(opt.n_best):
            if (i+1) in print_topk:
                if total_chirality > 0:
                    print("Top-{} Accuracy with chirality:{:.3f}%".format(i + 1, topn_accuracy_chirality[i] / total_chirality * 100))
                    save_dict[f'top-{i+1}_chilarity'] = topn_accuracy_chirality[i] / total_chirality * 100
                print("Top-{} Accuracy without chirality:{:.3f}%".format(i + 1, topn_accuracy_wochirality[i] / (data_size - total_chirality) * 100))
                save_dict[f'top-{i+1}_wochilarity'] = topn_accuracy_wochirality[i] / (data_size - total_chirality) * 100
                if total_ringopening > 0:
                    print("Top-{} Accuracy ring Opening:{:.3f}%".format(i + 1, topn_accuracy_ringopening[i] / total_ringopening * 100))
                    save_dict[f'top-{i+1}_ringopening'] = topn_accuracy_ringopening[i] / total_ringopening * 100
                if total_ringformation > 0:
                    print("Top-{} Accuracy ring Formation:{:.3f}%".format(i + 1, topn_accuracy_ringformation[i] / total_ringformation * 100))
                    save_dict[f'top-{i+1}_ringformation'] = topn_accuracy_ringformation[i] / total_ringformation * 100
                print("Top-{} Accuracy without ring:{:.3f}%".format(i + 1, topn_accuracy_woring[i] / (data_size - total_ringopening - total_ringformation) * 100))
                save_dict[f'top-{i+1}_wocring'] = topn_accuracy_woring[i] /  (data_size - total_ringopening - total_ringformation)* 100
        print(total_chirality)
        print(total_ringformation)
        print(total_ringopening)
        # df = pd.DataFrame(list(save_dict.items()))
        df = pd.DataFrame(save_dict,index=[0])
        df.to_csv("detailed_results.csv")
    if opt.save_accurate_indices != "":
        with open(opt.save_accurate_indices, "w") as f:
            total_accurate_indices = []
            for indices in accurate_indices:
                total_accurate_indices.extend(indices)
            total_accurate_indices.sort()

            # for index in total_accurate_indices:
            for index in accurate_indices[0]:
                f.write(str(index))
                f.write("\n")

    if opt.save_file != "":
        with open(opt.save_file,"w") as f:
            for res in ranked_results:
                for smi in res:
                    f.write(smi)
                    f.write("\n")
                for i in range(len(res),opt.n_best):
                    f.write("")
                    f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beam_size', type=int, default=10,help='Beam size')
    parser.add_argument('--n_best', type=int, default=10,help='n best')
    parser.add_argument('--predictions', type=str, required=True,
                        help="Path to file containing the predictions")
    parser.add_argument('--targets', type=str, required=True, help="Path to file containing targets")
    parser.add_argument('--sources', type=str, default="", help="Path to file containing sources")
    parser.add_argument('--augmentation', type=int, default=20)
    parser.add_argument('--score_alpha', type=float, default=1.0)
    parser.add_argument(
        '--aggregation_mode',
        choices=[
            "legacy_best_rank",
            "rrf",
            "frequency_first",
            "hybrid",
        ],
        default="legacy_best_rank",
        help="Cross-augmentation candidate aggregation rule",
    )
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument(
        '--target_offset',
        type=int,
        default=0,
        help="0-based original-reaction offset in targets/sources",
    )
    parser.add_argument('--process_number', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--synthon', action="store_true", default=False)
    parser.add_argument('--detailed', action="store_true", default=False)
    parser.add_argument('--raw', action="store_true", default=False)
    parser.add_argument('--save_file', type=str,default="")
    parser.add_argument('--save_accurate_indices', type=str,default="")
    parser.add_argument(
        '--diagnostics',
        action="store_true",
        default=False,
        help=(
            "Report oracle coverage, per-run validity/diversity, and "
            "cross-run overlap without changing aggregation"
        ),
    )
    parser.add_argument(
        '--diagnostics_json',
        type=str,
        default="",
        help="Optional path for machine-readable per-reaction diagnostics",
    )

    opt = parser.parse_args()
    print(opt)
    main(opt)

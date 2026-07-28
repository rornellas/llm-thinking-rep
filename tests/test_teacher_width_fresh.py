from pathlib import Path

import yaml

from pre_qwen_certification.controlled_transplant import _tiny_config
from pre_qwen_certification.data import jaccard, text_sha256, word_shingles
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.teacher_width_data import generate_width_documents, generate_width_ood_documents
from pre_qwen_certification.tiny_lm import TinyMoELanguageModel, make_narrow_student

ROOT = Path(__file__).resolve().parents[1]


def test_width_candidates_have_exact_ratios_and_frozen_router():
    config = yaml.safe_load((ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml").read_text())
    tiny = _tiny_config(config["model"])
    teacher = TinyMoELanguageModel(97, tiny).blocks[0].moe
    assert isinstance(teacher, ConventionalSwiGLUMoE)
    for width in config["widths"].values():
        student = make_narrow_student(teacher, d_ff=int(width))
        assert student.geometry.d_ff / teacher.geometry.d_ff == int(width) / 40.0
        assert not student.router.weight.requires_grad


def test_fresh_document_families_have_no_exact_or_high_shingle_overlap():
    config = yaml.safe_load((ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml").read_text())
    data = config["data"]
    groups = [
        generate_width_documents(split=data["train_split"], documents=data["train_documents"], seed=data["train_seed"]),
        generate_width_documents(split=data["hypothesis_split"], documents=data["hypothesis_documents"], seed=data["hypothesis_seed"]),
        generate_width_ood_documents(split=data["ood_split"]),
    ]
    all_docs = [doc for group in groups for doc in group]
    assert len({text_sha256(doc.text) for doc in all_docs}) == len(all_docs)
    maximum = 0.0
    for left_index, left_group in enumerate(groups):
        for right_group in groups[left_index + 1:]:
            for left in left_group:
                for right in right_group:
                    maximum = max(maximum, jaccard(word_shingles(left.text), word_shingles(right.text)))
    assert maximum < float(data["near_duplicate_threshold"])


def test_frozen_train_vocabulary_covers_hypothesis_and_ood():
    config = yaml.safe_load((ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml").read_text())
    data = config["data"]
    train = generate_width_documents(split=data["train_split"], documents=data["train_documents"], seed=data["train_seed"])
    hypothesis = generate_width_documents(split=data["hypothesis_split"], documents=data["hypothesis_documents"], seed=data["hypothesis_seed"])
    ood = generate_width_ood_documents(split=data["ood_split"])
    train_chars = set("".join(doc.text for doc in train))
    assert set("".join(doc.text for doc in hypothesis + ood)) <= train_chars

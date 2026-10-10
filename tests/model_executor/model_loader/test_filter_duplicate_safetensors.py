# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile

import pytest

from vllm import envs
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)


def test_filter_duplicate_safetensors_files_missing_weight():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        with open(existing_file, "wb") as f:
            f.write(b"")

        existing_file2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        with open(existing_file2, "wb") as f:
            f.write(b"")

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
                "layer.2.weight": "model-00003-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        hf_weights_files = [
            os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
        ]

        with pytest.raises(FileNotFoundError) as exc_info:
            filter_duplicate_safetensors_files(
                hf_weights_files=hf_weights_files,
                hf_folder=tmpdir,
                index_file="model.safetensors.index.json",
            )

        assert "model-00003-of-00002.safetensors" in str(exc_info.value)


def test_filter_duplicate_safetensors_files_all_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_files = []
        for i in range(1, 3):
            file_path = os.path.join(tmpdir, f"model-0000{i}-of-00002.safetensors")
            with open(file_path, "wb") as f:
                f.write(b"")
            existing_files.append(file_path)

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        filter_duplicate_safetensors_files(
            hf_weights_files=existing_files,
            hf_folder=tmpdir,
            index_file="model.safetensors.index.json",
        )


@pytest.mark.parametrize(
    "cpu_offload,quant_dir,extra_weight,allowed",
    [
        (True, "sidecar", None, True),
        (False, "sidecar", None, False),
        (True, "", None, False),
        (False, "", None, False),
        (True, "sidecar", "layer.0.weight", False),
    ],
)
def test_missing_ple_shard_requires_sidecar_and_ple_only_contents(
    tmp_path, monkeypatch, cpu_offload, quant_dir, extra_weight, allowed
):
    """A sidecar replaces PLE-only shards, not other checkpoint weights."""
    monkeypatch.setattr(envs, "VLLM_PLE_CPU_OFFLOAD", cpu_offload)
    monkeypatch.setattr(envs, "VLLM_PLE_QUANT_DIR", quant_dir)
    existing = tmp_path / "backbone.safetensors"
    existing.touch()
    missing = "ple.safetensors"
    weight_map = {
        "layer.1.weight": existing.name,
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": missing,
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight": missing,
    }
    if extra_weight is not None:
        weight_map[extra_weight] = missing
    index_file = "model.safetensors.index.json"
    (tmp_path / index_file).write_text(json.dumps({"weight_map": weight_map}))
    files = [str(existing)]

    if allowed:
        assert (
            filter_duplicate_safetensors_files(files, str(tmp_path), index_file)
            == files
        )
    else:
        with pytest.raises(FileNotFoundError, match=missing):
            filter_duplicate_safetensors_files(files, str(tmp_path), index_file)


def test_sidecar_does_not_hide_missing_backbone_shard(tmp_path, monkeypatch):
    """Ignoring a PLE shard must not suppress another missing-file error."""
    monkeypatch.setattr(envs, "VLLM_PLE_CPU_OFFLOAD", True)
    monkeypatch.setattr(envs, "VLLM_PLE_QUANT_DIR", "sidecar")
    index_file = "model.safetensors.index.json"
    weight_map = {
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": (
            "ple.safetensors"
        ),
        "layer.0.weight": "backbone.safetensors",
    }
    (tmp_path / index_file).write_text(json.dumps({"weight_map": weight_map}))

    with pytest.raises(FileNotFoundError, match="backbone.safetensors") as exc:
        filter_duplicate_safetensors_files([], str(tmp_path), index_file)
    assert "ple.safetensors" not in str(exc.value)


if __name__ == "__main__":
    test_filter_duplicate_safetensors_files_missing_weight()
    test_filter_duplicate_safetensors_files_all_exist()

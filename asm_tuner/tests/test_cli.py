import json

from asm_tuner.cli import main
from asm_tuner.schema import DType
from asm_tuner.vendor_fixtures import default_seed


def test_fixture_and_seed_commands_are_phase_independent(capsys):
    assert main(["fixture", "bf16_solution205554"]) == 0
    fixture = json.loads(capsys.readouterr().out)
    assert (fixture["bm"], fixture["bn"]) == (128, 224)
    assert fixture["derived"]["miwt"] == {"m": 4, "n": 7}

    assert main(["seed", "--adapter", "bf16"]) == 0
    seed = json.loads(capsys.readouterr().out)
    assert seed["provenance"]["vendor_default"] is False


def test_validate_and_prune_commands_use_only_explicit_files(tmp_path, capsys):
    config_path = tmp_path / "configs.json"
    route_path = tmp_path / "routes.json"
    config_path.write_text(json.dumps({"configs": [default_seed(DType.BF16).to_dict()]}))
    route_path.write_text(json.dumps({"routes": [{"name": "tiny", "m": 1, "n": 17, "k": 32}]}))

    assert main(["validate", "--configs", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["reports"][0]["valid"] is True
    assert main(["prune", "--configs", str(config_path), "--routes", str(route_path), "--cu-count", "304"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["grid_policy"] == "sort_only_never_hard_prune"
    assert result["shape_ranks"]["tiny"][0]["grid"][2] == 1

import sys

from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/home/prasoon/Documents/research/flame")
import custom_models  # noqa: E402,F401


def test_msgla_scale_configs_load():
    config_paths = [
        "/home/prasoon/Documents/research/flame/configs/ms_gla_340M_s12.json",
        "/home/prasoon/Documents/research/flame/configs/ms_gla_340M_s24.json",
        "/home/prasoon/Documents/research/flame/configs/ms_gla_340M.json",
        "/home/prasoon/Documents/research/flame/configs/ms_gla_340M_s1248.json",
    ]

    expected = {
        config_paths[0]: ([1, 2], [2, 2]),
        config_paths[1]: ([2, 4], [2, 2]),
        config_paths[2]: ([1, 2, 4], [2, 1, 1]),
        config_paths[3]: ([1, 2, 4, 8], [1, 1, 1, 1]),
    }

    for config_path in config_paths:
        cfg = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(cfg)
        expected_scales, expected_heads = expected[config_path]

        assert cfg.model_type == "ms_gla"
        assert cfg.scales == expected_scales
        assert cfg.scale_num_heads == expected_heads
        assert type(model).__name__ == "MSGLAForCausalLM"

import sys
import types

sys.path.insert(0, "/home/prasoon/Documents/research/flame")

import custom_models
from transformers import AutoConfig, AutoModelForCausalLM

config_manager = types.ModuleType("torchtitan.config_manager")
config_manager.TORCH_DTYPE_MAP = {}
config_manager.JobConfig = object
parallel_dims = types.ModuleType("torchtitan.distributed.parallel_dims")
parallel_dims.ParallelDims = object
logging_mod = types.ModuleType("torchtitan.tools.logging")
logging_mod.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

sys.modules.setdefault("torchtitan", types.ModuleType("torchtitan"))
sys.modules["torchtitan.config_manager"] = config_manager
sys.modules.setdefault("torchtitan.distributed", types.ModuleType("torchtitan.distributed"))
sys.modules["torchtitan.distributed.parallel_dims"] = parallel_dims
sys.modules.setdefault("torchtitan.tools", types.ModuleType("torchtitan.tools"))
sys.modules["torchtitan.tools.logging"] = logging_mod

from flame.models.parallelize_fla import MSGLATPPlan, TP_PLAN_MAP


def test_msgla_tp_plan_registration():
    assert TP_PLAN_MAP["ms_gla"] is MSGLATPPlan


def test_msgla_tp_plan_contains_branch_parallelism():
    cfg = AutoConfig.for_model(
        "ms_gla",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_heads=4,
        scales=[1, 2, 4],
        scale_num_heads=[2, 1, 1],
        expand_k=0.5,
        expand_v=1.0,
        vocab_size=128,
        fuse_cross_entropy=False,
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)

    plan = MSGLATPPlan(model)
    attn_plan = plan.attn_plan

    assert "attn" in attn_plan
    assert "attn.fuse" in attn_plan

    for branch_idx in range(len(cfg.scales)):
        branch_prefix = f"attn.branches.{branch_idx}"
        assert f"{branch_prefix}.q_proj" in attn_plan
        assert f"{branch_prefix}.k_proj" in attn_plan
        assert f"{branch_prefix}.v_proj" in attn_plan
        assert f"{branch_prefix}.g_proj" in attn_plan
        assert f"{branch_prefix}.gk_proj.0" in attn_plan
        assert f"{branch_prefix}.gk_proj.1" in attn_plan
        assert f"{branch_prefix}.g_norm" in attn_plan
        assert f"{branch_prefix}.o_proj" in attn_plan

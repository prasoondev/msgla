from __future__ import annotations

from typing import Iterable

from fla.models.gla.configuration_gla import GLAConfig


def _normalize_scale_num_heads(
    num_heads: int,
    scales: list[int],
    scale_num_heads: list[int] | None,
) -> list[int]:
    if scale_num_heads is not None:
        if len(scale_num_heads) != len(scales):
            raise ValueError("`scale_num_heads` must have the same length as `scales`.")
        if any(heads <= 0 for heads in scale_num_heads):
            raise ValueError("Each entry in `scale_num_heads` must be positive.")
        if sum(scale_num_heads) != num_heads:
            raise ValueError("`scale_num_heads` must sum to `num_heads` to preserve the total state budget.")
        return list(scale_num_heads)

    if num_heads < len(scales):
        raise ValueError("`num_heads` must be at least the number of scales when `scale_num_heads` is omitted.")

    base = num_heads // len(scales)
    remainder = num_heads % len(scales)
    heads = [base] * len(scales)
    for i in range(remainder):
        heads[i] += 1
    return heads


class MSGLAConfig(GLAConfig):
    model_type = "ms_gla"

    def __init__(
        self,
        scales: Iterable[int] = (1, 2, 4),
        scale_num_heads: list[int] | None = None,
        pool_mode: str = "avg",
        fuse_mode: str = "softmax",
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(use_cache=use_cache, **kwargs)

        self.scales = [int(scale) for scale in scales]
        if len(self.scales) == 0:
            raise ValueError("`scales` must contain at least one scale.")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("All entries in `scales` must be positive integers.")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("`scales` must not contain duplicates.")

        if pool_mode not in {"avg"}:
            raise ValueError("Only `avg` pooling is currently supported for MS-GLA.")
        if fuse_mode not in {"softmax", "mean"}:
            raise ValueError("`fuse_mode` must be either `softmax` or `mean`.")

        if self.num_kv_heads is not None and self.num_kv_heads != self.num_heads:
            raise ValueError(
                "MS-GLA currently expects `num_kv_heads` to be unset or equal to `num_heads` "
                "so the head budget can be split cleanly across scales."
            )

        self.scale_num_heads = _normalize_scale_num_heads(self.num_heads, self.scales, scale_num_heads)
        self.pool_mode = pool_mode
        self.fuse_mode = fuse_mode

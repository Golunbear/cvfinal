"""Conditional Flow Matching: from LR to HR (residual prediction)."""

from __future__ import annotations

import torch
import torch.nn as nn


def sample_lr_to_hr_path(
    condition: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_view = t[:, None, None, None]
    x_t = (1.0 - t_view) * condition + t_view * target
    u_t = target - condition
    return x_t, u_t


def flow_matching_loss(
    model: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    b = target.shape[0]
    t = torch.rand(b, device=target.device, dtype=target.dtype)
    x_t, u_t = sample_lr_to_hr_path(condition, target, t)
    v_pred = model(x_t, t, condition)
    return torch.nn.functional.mse_loss(v_pred, u_t)


@torch.inference_mode()
def euler_solve(
    model: nn.Module,
    condition: torch.Tensor,
    steps: int = 25,
    fast: bool = False,
) -> torch.Tensor:
    b, c, h, w = condition.shape
    x = condition
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=condition.device, dtype=condition.dtype)
        if fast:
            v = model(x, t, condition, fast=True)
        else:
            v = model(x, t, condition)
        x = x + v * dt
    return x


@torch.inference_mode()
def midpoint_solve(
    model: nn.Module,
    condition: torch.Tensor,
    steps: int = 25,
) -> torch.Tensor:
    b, c, h, w = condition.shape
    x = condition
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=condition.device, dtype=condition.dtype)
        v1 = model(x, t, condition)
        x_mid = x + v1 * (dt / 2)
        t_mid = torch.full((b,), t_val + dt / 2, device=condition.device, dtype=condition.dtype)
        v2 = model(x_mid, t_mid, condition)
        x = x + v2 * dt
    return x


@torch.inference_mode()
def deterministic_solve(
    model: nn.Module,
    condition: torch.Tensor,
    noise: torch.Tensor,
    steps: int = 25,
    solver: str = "euler",
) -> torch.Tensor:
    b = condition.shape[0]
    x = condition
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=condition.device, dtype=condition.dtype)
        v = model(x, t, condition)
        if solver == "midpoint" and i < steps - 1:
            x_mid = x + v * (dt / 2)
            t_mid = torch.full((b,), t_val + dt / 2, device=condition.device, dtype=condition.dtype)
            v = model(x_mid, t_mid, condition)
        x = x + v * dt
    return x

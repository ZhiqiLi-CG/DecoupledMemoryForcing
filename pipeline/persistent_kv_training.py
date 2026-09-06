"""One-pass Self Gradient Forcing via a persistent differentiable KV write.

SGF spends two full-clip forwards per step: an autoregressive rollout under
no_grad, then one parallel teacher-forced forward that carries the gradient.
This pipeline produces the same gradient in one pass, by making the rollout
itself differentiable.

The trick is that each frame's memory write returns immutable K/V pages instead
of overwriting a cache buffer. Nothing is detached, so frame i's loss reaches
frame j<i through the page chain; and because the pages never change, each
frame's forward can be activation-checkpointed and recomputed at backward time
against exactly the inputs it originally saw. A mutated cache would fail that
recomputation, which is why every earlier one-pass method had to cut the
cross-frame gradient instead.

Equivalence to SGF's parallel teacher forcing is verified, not assumed:
the global gradient vector agrees to 1.75e-04 with cosine 0.99999988, while the
two attention kernels involved disagree by 3.6e-03 (forward) / 5.4e-03
(backward) on identical inputs -- so the two paths differ by less than the
kernels they are built on. See docs/persistent_kv_write.md.

Structurally identical to _self_gradient_forcing_inference: same exit-step
sampling, same context noising, same x0 caching modes. The only change is that
the second pass is gone and its work happens inside the rollout.
"""
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint

from wan.modules.persistent_kv_model import PersistentKVStore


class PersistentKVTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: torch.Tensor,
        scheduler,
        generator,
        num_frame_per_block=1,
        num_max_frames=21,
        context_noise=0,
        per_rank_exit_step: bool = True,
        self_gradient_forcing_match_context: bool = True,
        self_gradient_forcing_cache_mode: str = "exit",
        gradient_checkpointing: bool = True,
        checkpoint_inside_model: bool = True,
        **kwargs,
    ):
        self.denoising_step_list = denoising_step_list
        # Same two constructor guards as SelfGradientForcingTrainingPipeline.
        # They are inert under the shipped config (warped list has no trailing
        # 0; cache_mode is 'exit') but without them a config change would make
        # the two pipelines diverge silently: the reference would drop a final
        # t=0 step and refuse an unknown cache mode, this one would run a 5th
        # denoise at t=0 / fall into an undefined hybrid.
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        if self_gradient_forcing_cache_mode not in ("exit", "final_clean"):
            raise ValueError(
                "self_gradient_forcing_cache_mode must be 'exit' or 'final_clean'. "
                f"Got {self_gradient_forcing_cache_mode!r}."
            )
        self.scheduler = scheduler
        self.generator = generator
        self.num_frame_per_block = num_frame_per_block
        self.num_max_frames = num_max_frames
        self.context_noise = context_noise
        self.per_rank_exit_step = per_rank_exit_step
        self.match_context = self_gradient_forcing_match_context
        self.cache_mode = self_gradient_forcing_cache_mode
        self.gradient_checkpointing = gradient_checkpointing
        # Where the checkpoint sits relative to the FSDP boundary. Inside is the
        # composition PyTorch supports; outside is what this pipeline did first,
        # and it is the prime suspect for the gradient corruption seen under
        # FSDP + MixedPrecision.
        self.checkpoint_inside_model = checkpoint_inside_model
        # honoured by generate_and_sync_list, as in the official pipeline
        self.last_step_only = kwargs.get("last_step_only", False)

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def _model(self):
        # unwrap FSDP/DDP if present; the persistent forwards are model methods
        m = self.generator.model
        return getattr(m, "module", m)

    def _to_bcfhw(self, x):
        """[B, F, C, H, W] -> [B, C, F, H, W], the layout the model patchifies."""
        return x.permute(0, 2, 1, 3, 4)

    # Both forwards go through the generator wrapper, never through
    # self._model directly. Under FSDP the wrapper IS the root module, and its
    # forward is where the root's parameters -- patch_embedding, time_embedding,
    # head -- get all-gathered and cast to bf16. Calling the inner model skips
    # that: single-GPU tests pass and multi-GPU training dies on
    # "Input type (c10::BFloat16) and bias type (float) should be the same".
    def _noisy(self, latent, timestep, cond, hist, start, ckpt):
        """One frame/block of the generation path. Reads history, writes nothing."""
        def fn(lat, ts, h):
            _, x0 = self.generator(
                noisy_image_or_video=lat, conditional_dict=cond, timestep=ts,
                persistent_mode="noisy", persistent_history=h,
                persistent_start=start)
            return x0
        return (checkpoint(fn, latent, timestep, hist, use_reentrant=False)
                if ckpt else fn(latent, timestep, hist))

    def _write(self, latent, timestep, cond, hist, start, ckpt):
        """The memory write. Returns pages flattened as (k0, v0, k1, v1, ...)."""
        def fn(lat, ts, h):
            return self.generator(
                noisy_image_or_video=lat, conditional_dict=cond, timestep=ts,
                persistent_mode="write", persistent_history=h,
                persistent_start=start)
        return (checkpoint(fn, latent, timestep, hist, use_reentrant=False)
                if ckpt else fn(latent, timestep, hist))

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        """Byte-for-byte the official version's behaviour.

        Drawing on every rank and then broadcasting gives the same index, but it
        advances the RNG on non-zero ranks where the official code does not, so
        every later per-rank draw diverges from SGF. It also silently ignored
        last_step_only. Neither shows up under the current config -- which is
        exactly why it would have surfaced as an unexplained difference during an
        ablation rather than as an error here.
        """
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.randint(low=0, high=num_denoising_steps,
                                        size=(num_blocks,), device=device)
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()
        indices = torch.randint(low=0, high=num_denoising_steps,
                                size=(num_blocks,), device=device)
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        return indices.tolist()

    # ── the pass ──────────────────────────────────────────────────────────
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ):
        assert initial_latent is None, \
            "Persistent-KV training assumes text-to-video inputs."
        batch_size, num_frames, _, _, _ = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device = noise.device
        model = self._model

        num_denoising_steps = len(self.denoising_step_list)
        if self.cache_mode == "exit" and self.per_rank_exit_step:
            idx = int(torch.randint(low=0, high=num_denoising_steps,
                                    size=(1,), device=device).item())
        else:
            idx = int(self.generate_and_sync_list(1, num_denoising_steps, device)[0])
        train_t = self.denoising_step_list[idx]

        # forward_persistent attends to EVERY page: it has no window and no sink
        # yet. That is exactly equivalent to the config's (sink, window) only
        # while the two together already cover the whole clip. At 21 frames with
        # sink=3, window=18 the masks are bit-identical (verified by
        # /mnt/data/zhaofeng/runs/sgf/tests/test_window_covers_all.py, kept
        # outside the repo with the other one-pass tests: 0 differing entries;
        # at 24 frames they differ in 192). Past that point training would silently run on a
        # different attention pattern, so refuse instead.
        # Chunkwise is untested, so it is refused rather than run. In SGF's
        # teacher-forcing mask a clean token sees the other clean tokens of its
        # OWN block (causal_model.py, context_ends = the block's end); here the
        # block is written in one call and intra-block visibility is whatever
        # forward_clean_writer does. At num_frame_per_block=1 the two cannot
        # differ, which is exactly why every green light so far says nothing
        # about num_frame_per_block>1.
        if self.num_frame_per_block != 1:
            raise ValueError(
                "persistent-KV is only verified for num_frame_per_block=1; got "
                "%d. Chunkwise intra-block visibility has no test against SGF's "
                "mask, so it would run silently on a different attention "
                "pattern." % self.num_frame_per_block)

        local = getattr(model, "local_attn_size", -1)
        sink = getattr(model, "sink_size", 0)
        if local != -1 and num_frames > local + sink:
            raise ValueError(
                "persistent-KV attends to all history, which matches "
                "local_attn_size=%d + sink_size=%d only up to %d frames; got %d. "
                "Implement the window/sink in forward_persistent first."
                % (local, sink, local + sink, num_frames))

        model.persistent_block_checkpointing = (self.gradient_checkpointing
                                                and self.checkpoint_inside_model)
        store = PersistentKVStore(len(model.blocks))
        outputs = []
        # when the model checkpoints internally, do not also checkpoint here
        ckpt = (self.gradient_checkpointing and torch.is_grad_enabled()
                and not self.checkpoint_inside_model)
        start = 0
        for _ in range(num_blocks):
            n = self.num_frame_per_block
            # Every page written so far. Snapshotting per block keeps the tuple
            # immutable for this block's checkpoint recomputation.
            hist = store.snapshot_all_layers(list(range(len(store.k_pages[0]))))

            # ── the rollout, no gradients (as in SGF pass 1) ──────────────
            with torch.no_grad():
                noisy_input = noise[:, start:start + n]
                noisy_at_t, x0_exit, timestep = None, None, None
                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = torch.ones([batch_size, n], device=device,
                                          dtype=torch.int64) * current_timestep
                    if index == idx:
                        noisy_at_t = noisy_input
                    x0 = self._noisy(noisy_input, timestep, conditional_dict,
                                     hist, start, ckpt=False)
                    if index == idx:
                        x0_exit = x0
                        if self.cache_mode == "exit" and not self.per_rank_exit_step:
                            break
                    if index < num_denoising_steps - 1:
                        next_t = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                            next_t * torch.ones([batch_size * n], device=device,
                                                dtype=torch.long),
                        ).unflatten(0, x0.shape[:2])
                if x0_exit is None:
                    raise RuntimeError("Failed to capture the exit state.")

                x0_cache = x0 if self.cache_mode == "final_clean" else x0_exit
                context_timestep = torch.ones_like(timestep) * self.context_noise
                x_ctx = self.scheduler.add_noise(
                    x0_cache.flatten(0, 1), torch.randn_like(x0_cache.flatten(0, 1)),
                    context_timestep.flatten(0, 1).long(),
                ).unflatten(0, x0_cache.shape[:2])

            # ── the differentiable part: one noisy forward at the sampled
            #    training timestep, and one memory write ────────────────────
            tf_t = torch.ones([batch_size, n], device=device,
                              dtype=torch.int64) * train_t
            outputs.append(self._noisy(noisy_at_t, tf_t, conditional_dict, hist,
                                       start, ckpt=ckpt))

            if self.match_context:
                write_x, write_t = x_ctx, context_timestep
            else:
                write_x, write_t = x0_cache, torch.zeros_like(context_timestep)
            store.append_all_layers(
                self._write(write_x, write_t, conditional_dict, hist, start,
                            ckpt=ckpt))
            start += n

        output = torch.cat(outputs, dim=1)

        # identical to SGF: the DMD objective needs the timestep window the
        # sampled exit step corresponds to
        ts = self.scheduler.timesteps.to(device)
        if idx == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (ts - self.denoising_step_list[idx].to(device)).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (ts - self.denoising_step_list[idx + 1].to(device)).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (ts - self.denoising_step_list[idx].to(device)).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, idx + 1
        return output, denoised_timestep_from, denoised_timestep_to

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import parallel_state as mpu

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.use_torch_profiler = self.config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(
                self.config.profiler, tool_config=self.config.profiler.get("tool_config", {}).get("torch", {})
            )
        else:
            self.prof = None
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if self.use_fused_kernels:
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)

        config = get_model_config(self.actor_module[0])
        if torch.distributed.get_rank() == 0:
            print(config)
        
        # Router logits configuration
        self.use_router_logits = self.config.get("use_router_logits", False)
        self.use_router_kl_loss = self.config.get("use_router_kl_loss", False)
        self.use_router_shift = self.config.get("use_router_shift", False)
        if torch.distributed.get_rank() == 0:
            print(f"Router logits collection enabled: {self.use_router_logits}")
            print(f"Router KL loss enabled: {self.use_router_kl_loss}")
            print(f"Router shift enabled: {self.use_router_shift}")
            if self.use_router_shift:
                print(f"Router shift clip threshold: {self.config.get('router_shift_clip_threshold', 0.0)}")
            if self.use_router_kl_loss and not self.use_router_logits:
                print("WARNING: Router KL loss is enabled but router logits collection is disabled!")
                print("Please set use_router_logits=True to enable router KL loss.")
            if self.use_router_kl_loss:
                print("NOTE: Router KL loss requires old_router_logits from previous iteration.")
                print("      First training iteration will skip router KL loss.")

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        entropys = torch.Tensor()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        input_ids = batch["input_ids"]
        batch_size = input_ids.size(0)
        response = batch["responses"]
        response_length = response.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(
                data,
                forward_only=True,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )

            # -------------------------
            # Collect log_probs / entropy (last PP stage only)
            # -------------------------
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                micro_outputs = output["output"]
                log_probs = [o["log_probs"] for o in micro_outputs]
                log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                if calculate_entropy:
                    entropys = torch.cat([o["entropy"] for o in micro_outputs], dim=0).to(torch.float32)
                if use_dynamic_bsz:
                    indices = output.get("indices", None)
                    if indices is not None:
                        indices_flat = list(itertools.chain.from_iterable(indices))
                        assert len(indices_flat) == log_probs.size(0), (
                            f"{len(indices_flat)} vs. {log_probs.size()}"
                        )
                        revert_indices = torch.tensor(get_reverse_idx(indices_flat), dtype=torch.long)
                        log_probs = log_probs[revert_indices]
                        if calculate_entropy:
                            assert len(indices_flat) == entropys.size(0), (
                                f"{len(indices_flat)} vs. {entropys.size()}"
                            )
                            entropys = entropys[revert_indices]
            else:
                log_probs = torch.empty(
                    size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                )
                if calculate_entropy:
                    entropys = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )

            pp_group = mpu.get_pipeline_model_parallel_group()
            pp_world_size = torch.distributed.get_world_size(pp_group)
            # -------------------------
            # Gather and aggregate router_logits across PP stages (only if enabled)
            # -------------------------
            if self.use_router_logits:
                # Local partial shape: [B_micro, S, L_local, E]; concat micro-batches on batch dim first.
                # Steps:
                #   1. Concat local parts -> local_router [B_total, S, L_local, E] (or None)
                #   2. Exchange (L_local, E) via all_gather (int tensor)
                #   3. Pad local_router to max_L along layer dim -> send_buf [B_total, S, max_L, E_global]
                #   4. all_gather padded tensors -> list; slice per-rank L_local and concat along layer dim
                #   5. (Optional) reorder batch if dynamic_bsz
                # Result: aggregated_router_logits [B_total, S, sum(L_local), E_global] or None
                
                local_router_parts: list[torch.Tensor] = []
                for mb_out in output["output"]:
                    if isinstance(mb_out, dict):
                        rl = mb_out.get("router_logits")
                        if rl is not None:
                            local_router_parts.append(rl)
                            # Clean up the reference in mb_out to save memory
                            del mb_out["router_logits"]
                
                local_router = torch.cat(local_router_parts, dim=0) if local_router_parts else None
                # Clean up the parts list
                del local_router_parts

                
                # ---------------- Router logits aggregation (keep log_probs/entropy logic unchanged) ----------------
                if pp_world_size == 1:
                    # Fast path: no communication for router logits
                    aggregated_router_logits = local_router
                    if (
                        aggregated_router_logits is not None
                        and use_dynamic_bsz
                        and mpu.is_pipeline_last_stage(ignore_virtual=True)
                        and indices is not None
                    ):
                        aggregated_router_logits = aggregated_router_logits[revert_indices]
                else:
                    # Meta gather for layer/expert dims
                    # Assumption: all ranks have identical expert dimension E (user confirmed).
                    # We only all_gather layer counts L_local; E obtained via all_reduce(max).
                    if local_router is not None:
                        B_total, S_total, L_local, E_global = local_router.shape
                    else:
                        B_total, S_total = input_ids.shape
                        L_local, E_global = 0, 0
                    
                    # Gather layer counts
                    L_local_tensor = torch.tensor([L_local], device=get_device_id(), dtype=torch.int32)
                    L_list_tensors = [torch.zeros_like(L_local_tensor) for _ in range(pp_world_size)]
                    torch.distributed.all_gather(L_list_tensors, L_local_tensor, group=pp_group)
                    L_list = [int(t[0].item()) for t in L_list_tensors]
                    
                    # All-reduce expert dim (max) to cover cases where some stages have no MoE layers
                    E_tensor = torch.tensor([E_global], device=get_device_id(), dtype=torch.int32)
                    torch.distributed.all_reduce(E_tensor, op=torch.distributed.ReduceOp.MAX, group=pp_group)
                    E_global = int(E_tensor.item())
                    L_max = max(L_list)

                    if sum(L_list) == 0 or E_global == 0:
                        aggregated_router_logits = None
                    else:
                        if local_router is None:
                            send_buf = torch.zeros(
                                (B_total, S_total, L_max, E_global), device=get_device_id(), dtype=torch.float32
                            )
                        else:
                            if local_router.dtype != torch.float32:
                                local_router = local_router.to(torch.float32)
                            send_buf = torch.zeros(
                                (B_total, S_total, L_max, E_global),
                                device=local_router.device,
                                dtype=local_router.dtype,
                            )
                            send_buf[:, :, :L_local, :E_global] = local_router
                        
                        send_buf_size_mb = send_buf.numel() * send_buf.element_size() / (1024**2)
                        gather_bufs = [torch.empty_like(send_buf) for _ in range(pp_world_size)]
                        torch.distributed.all_gather(gather_bufs, send_buf, group=pp_group)

                        # Clean up send_buf immediately after all_gather
                        del send_buf
                        
                        layer_segments = []
                        for rank_idx, buf in enumerate(gather_bufs):
                            l_len = L_list[rank_idx]
                            if l_len > 0:
                                layer_segments.append(buf[:, :, :l_len, :E_global])
                        
                        # Clean up gather_bufs after extracting segments
                        del gather_bufs
                        
                        aggregated_router_logits = torch.cat(layer_segments, dim=2) if layer_segments else None
                        
                        # Clean up layer_segments
                        del layer_segments

                        if (
                            aggregated_router_logits is not None
                            and use_dynamic_bsz
                            and mpu.is_pipeline_last_stage(ignore_virtual=True)
                            and indices is not None
                        ):
                            aggregated_router_logits = aggregated_router_logits[revert_indices]
                        

            else:
                # Router logits collection disabled
                aggregated_router_logits = None


            # ---------------- Broadcast (unchanged for log_probs / entropy) ----------------
            log_probs = log_probs.to(get_device_id())
            torch.distributed.broadcast(
                tensor=log_probs,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=pp_group,
                async_op=False,
            )
            log_probs = log_probs.to("cpu")

            if calculate_entropy:
                entropys = entropys.to(get_device_id())
                torch.distributed.broadcast(
                    tensor=entropys,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=pp_group,
                    async_op=False,
                )
                entropys = entropys.to("cpu")

            # Router logits broadcast - only if enabled
            if self.use_router_logits and aggregated_router_logits is not None:
                aggregated_router_logits = aggregated_router_logits.to(get_device_id())
                torch.distributed.broadcast(
                    tensor=aggregated_router_logits,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=pp_group,
                    async_op=False,
                )
                aggregated_router_logits = aggregated_router_logits.to("cpu")
            elif not self.use_router_logits:
                aggregated_router_logits = None

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return log_probs, entropys, aggregated_router_logits

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Add old_router_logits only if router-based features are actually used
        if self.config.get("use_router_logits", False) and (
            self.config.get("use_router_shift", False) or 
            self.config.get("use_router_kl_loss", False)
        ):
            select_keys.append("old_router_logits")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def compute_ppo_loss(self, model_output, data):
        log_prob = model_output["log_probs"]
        entropy = model_output.get("entropy", None)
        # get router logits
        router_logits = model_output.get("router_logits", None)
        metrics = {}

        response_mask = data["response_mask"].to(bool)
        # compute policy loss
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]

        # get old router logits - should now be directly available in data after minibatch processing
        old_router_logits = data.get("old_router_logits", None)
        
        # Memory optimization: only move to GPU and keep in memory when actually needed
        if old_router_logits is not None:
            need_router_computation = (
                self.config.get("use_router_shift", False) or 
                self.config.get("use_router_kl_loss", False)
            )
            
            if need_router_computation:
                # Move to GPU only when needed for computation
                old_router_logits = old_router_logits.to(log_prob.device)
            else:
                # If not needed for computation, remove immediately to save memory
                old_router_logits = None
                
            # Remove from data dict to free memory immediately after extraction
            if "old_router_logits" in data:
                del data["old_router_logits"]

        loss_agg_mode = self.config.loss_agg_mode

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        # add router ratio - memory optimized version with improved numerical stability
        router_shift_geometric_mean = None
        router_clip_mask = None
        if self.config.get("use_router_shift", False) and old_router_logits is not None and router_logits is not None:
            # Memory optimization: avoid creating full softmax matrices and use streaming computation
            with torch.no_grad():  # Router shift calculation doesn't need gradients
                eps = 1e-8
                
                # Step 1: First compute full softmax for both old and current router logits
                # This ensures we're working with proper probability distributions
                old_router_probs = F.softmax(old_router_logits, dim=-1)
                current_router_probs = F.softmax(router_logits, dim=-1)
                
                # Step 2: Get top-k experts from old router probabilities for efficient computation
                k_experts = min(8, old_router_logits.size(-1))  # Adaptive k based on available experts
                old_topk_probs, topk_indices = torch.topk(old_router_probs, k_experts, dim=-1)
                
                # Step 3: Extract corresponding probabilities from current router
                selected_current_probs = torch.gather(current_router_probs, dim=-1, index=topk_indices)
                
                # Clean up full probability matrices to save memory
                del old_router_probs, current_router_probs
                
                # Step 4: Compute shift using exp(-|Δlog p|) for numerical stability
                # Convert probabilities to log space for stable computation
                old_log_probs = torch.log(torch.clamp(old_topk_probs, min=eps))
                current_log_probs = torch.log(torch.clamp(selected_current_probs, min=eps))
     
                # Calculate |Δlog p| = |log(p_new) - log(p_old)|
                delta_log_probs = torch.abs(current_log_probs - old_log_probs)
                
                # Compute router shift as exp(-|Δlog p|) for each expert
                router_shift = torch.exp(-delta_log_probs)
                
                # Clean up intermediate tensors
                del old_topk_probs, selected_current_probs, old_log_probs, current_log_probs, delta_log_probs
                
                # Step 5: Streaming aggregation to minimize peak memory usage
                router_shift_mean = router_shift.mean(dim=-1)  # [bsz, seq_len, num_layers] experts level mean
                del router_shift  # Clean up after aggregation
                
                # Step 6: Log-space geometric mean calculation for numerical stability across layers
                log_shift = torch.log(torch.clamp(router_shift_mean, min=eps))
                del router_shift_mean  # Clean up after log computation
                if self.config.get("router_shift_ratio_geo_mean", False):
                    log_geom_mean = log_shift.mean(dim=-1)  # [bsz, seq_len] - average over layers
                else:
                    log_geom_mean = log_shift.sum(dim=-1)  # [bsz, seq_len] - 层累乘，但不做几何平均
                router_shift_geometric_mean = torch.exp(log_geom_mean)  # [bsz, seq_len]

                # Create clip mask: tokens below threshold need to be clipped
                clip_threshold = self.config.get("router_shift_clip_threshold", 0.7)
                router_clip_mask = (router_shift_geometric_mean < clip_threshold) & response_mask
                
        # Calculate router shift ratio only if we actually computed it
        if router_shift_geometric_mean is not None:
            # Compute clip fraction: count tokens that need clipping
            import verl.utils.torch_functional as verl_F
            if router_clip_mask is not None:
                router_shift_clipfrac = verl_F.masked_mean(
                    router_clip_mask.float(),
                    response_mask
                )
                metrics["actor/router_shift_clipfrac"] = router_shift_clipfrac.detach().item()
            
            # Compute mean router shift ratio using aggregation function
            router_shift_ratio = agg_loss(loss_mat=router_shift_geometric_mean, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
            metrics["actor/router_shift_ratio"] = router_shift_ratio.detach().item()
            
            # Compute additional statistics: max and min values on valid tokens only
            valid_values = router_shift_geometric_mean[response_mask]
            
            if valid_values.numel() > 0:  # Ensure we have valid values
                router_shift_max = valid_values.max().detach().item()
                router_shift_min = valid_values.min().detach().item()
                
                metrics["actor/router_shift_ratio_max"] = router_shift_max
                metrics["actor/router_shift_ratio_min"] = router_shift_min
            
            # No extra temporaries to clean here

        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, overlap_frac = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            use_router_shift=self.config.get("use_router_shift", False),
            router_shift_geometric_mean=router_shift_geometric_mean if self.config.get("use_router_shift", False) else None,
            router_clip_mask=router_clip_mask if self.config.get("use_router_shift", False) else None,
            config=self.config,
        )
        
        # Clean up router shift tensor after policy loss calculation
        if router_shift_geometric_mean is not None:
            del router_shift_geometric_mean
        if router_clip_mask is not None:
            del router_clip_mask

        metrics.update(
            {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                "actor/overlap_frac": overlap_frac.detach().item(),
            }
        )
        policy_loss = pg_loss

        # add entropy loss
        if entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            entropy_coeff = self.config.entropy_coeff
            policy_loss -= entropy_coeff * entropy_loss

        # add kl loss
        if self.config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            # compute kl loss
            if ref_log_prob is not None:
                kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                if kld is not None:
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                    policy_loss += kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef
                else:
                    if torch.distributed.get_rank() == 0:
                        print("Warning: kl_penalty returned None, skipping KL loss")
            else:
                if torch.distributed.get_rank() == 0:
                    print("Warning: ref_log_prob is None, skipping KL loss")
        

        # add router kl loss
        if self.config.use_router_kl_loss:
            if router_logits is not None and old_router_logits is not None:
                # Memory optimized router KL computation with chunked processing for large tensors
                if old_router_logits.numel() > 1e8:  # If tensor is very large (>100M elements)
                    # Process in chunks to avoid OOM
                    batch_size = old_router_logits.size(0)
                    chunk_size = max(1, batch_size // 4)
                    kl_chunks = []
                    
                    for i in range(0, batch_size, chunk_size):
                        end_idx = min(i + chunk_size, batch_size)
                        old_chunk = old_router_logits[i:end_idx]
                        router_chunk = router_logits[i:end_idx]
                        
                        router_probs_chunk = F.log_softmax(router_chunk, dim=-1)
                        old_router_probs_chunk = F.log_softmax(old_chunk, dim=-1)
                        
                        kl_chunk = kl_penalty(logprob=router_probs_chunk, ref_logprob=old_router_probs_chunk, kl_penalty="low_var_kl")
                        if kl_chunk is not None:
                            kl_chunks.append(kl_chunk.cpu())  # Move to CPU to save GPU memory
                        
                        # Clean up chunk tensors immediately
                        del old_chunk, router_chunk, router_probs_chunk, old_router_probs_chunk, kl_chunk
                    
                    # Clean up original tensors
                    del old_router_logits
                    
                    # Concatenate results back on GPU
                    if kl_chunks:
                        router_kl = torch.cat(kl_chunks, dim=0).to(router_logits.device)
                        del kl_chunks
                    else:
                        router_kl = None
                else:
                    # Standard processing for smaller tensors
                    router_probs = F.log_softmax(router_logits, dim=-1)
                    old_router_probs = F.log_softmax(old_router_logits, dim=-1)
                    
                    # Clean up old_router_logits immediately after creating log_softmax
                    del old_router_logits
                    
                    router_kl = kl_penalty(logprob=router_probs, ref_logprob=old_router_probs, kl_penalty="low_var_kl")
                    
                    # Clean up intermediate tensors
                    del router_probs, old_router_probs
                
                if router_kl is not None:
                    router_kl_aggregated = router_kl.mean(dim=(2, 3))  # Average over layers and experts
                    
                    # Clean up router_kl after aggregation
                    del router_kl
                    
                    router_kl_loss = agg_loss(loss_mat=router_kl_aggregated, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                    policy_loss = policy_loss + router_kl_loss * self.config.router_kl_loss_coef
                    metrics["actor/router_kl_loss"] = router_kl_loss.detach().item()
                    metrics["actor/router_kl_coef"] = self.config.router_kl_loss_coef
                    
                    # Clean up router_kl_aggregated after use
                    del router_kl_aggregated
                else:
                    if torch.distributed.get_rank() == 0:
                        print("Warning: router kl_penalty returned None, skipping router KL loss")
            else:
                if torch.distributed.get_rank() == 0:
                    print("Warning: router_logits or old_router_logits is None, skipping router KL loss")
        
        # Aggressive final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force synchronization to ensure memory is actually freed
            torch.cuda.synchronize()
            
        # Clean up any remaining local variables that might hold tensor references
        import gc
        gc.collect()
            
        return policy_loss, metrics

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        
        # data.to(get_device_id())
        # data.batch = data.batch.contiguous()
        mini_batch = data
        # broadcast_dict_tensor(
        #     mini_batch.batch,
        #     src=mpu.get_pipeline_model_parallel_last_rank(),
        #     group=mpu.get_pipeline_model_parallel_group(),
        # )
        # mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()
        

        def loss_func(output, data):
            
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            device = output["log_probs"].device

            responses = data["responses"]
            response_length = responses.size(1)

            log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            model_output = {"log_probs": log_prob}
            if self.use_router_logits:
                router_logits = output["router_logits"][:, -response_length - 1 : -1].contiguous()
                model_output["router_logits"] = router_logits
            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                model_output["entropy"] = entropy

            if forward_only:
                # for inference
                return torch.tensor(1.0, device=device), model_output

            # for training
            # note that this loss function can be swapped with other loss functions such as SFT
            policy_loss, metrics = self.compute_ppo_loss(model_output, data)

            # return loss and stats
            
            return policy_loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                for key in batch["multi_modal_inputs"][0].keys():
                    idxs = batch["multi_modal_inputs_idx"]
                    mmi = batch["multi_modal_inputs"]
                    multi_modal_inputs[key] = torch.cat(
                        [mmi[idx].get(key) for idx in idxs if mmi[idx].get(key) is not None], dim=0
                    )
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                # return dict of [logits, entropy]
                output = forward_fn(
                    model,
                    input_ids,
                    position_ids,
                    attention_mask,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    use_router_logits=self.use_router_logits,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        if torch.distributed.get_rank() == 0:
                            logger.warning_once(
                                "For memory-efficient computation, enable fused kernels via "
                                "`actor_rollout_ref.model.use_fused_kernels=True`. "
                                "The current `clone()` operation ensures correctness but increases memory usage."
                            )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                
                output = forward_fn(
                    model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    use_router_logits=self.use_router_logits,
                )
                

            return output, partial(loss_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
        
        # Memory monitoring for GPU0 OOM debugging
        initial_memory = get_torch_device().memory_allocated() / (1024**3)
        if torch.distributed.get_rank() == 0:
            logger.info(f"update_policy starting - GPU memory: {initial_memory:.2f}GB")
        
        minibatch_count = 0
        for data in dataloader:
            minibatch_count += 1
            
            # Monitor memory before processing each minibatch (GPU0 OOM debugging)
            if torch.distributed.get_rank() == 0:
                current_memory = get_torch_device().memory_allocated() / (1024**3)
                memory_growth = current_memory - initial_memory
                logger.info(f"Minibatch {minibatch_count} - GPU memory: {current_memory:.2f}GB (growth: +{memory_growth:.2f}GB)")
                
                # Check for old_router_logits in the data
                if "old_router_logits" in data.batch:
                    router_size_mb = data.batch["old_router_logits"].numel() * data.batch["old_router_logits"].element_size() / (1024**2)
                    logger.info(f"Minibatch {minibatch_count} contains router_logits: {router_size_mb:.1f}MB")
            
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            if self.use_torch_profiler and self.prof and self.prof.enable:
                self.prof.step()
                
            # Aggressive memory cleanup after each minibatch to prevent memory accumulation
            get_torch_device().empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all operations complete before freeing memory
        # add empty cache after each compute
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.stop_and_save()
            self.prof.stop_trace()
        get_torch_device().empty_cache()
        
        # Final memory monitoring
        final_memory = get_torch_device().memory_allocated() / (1024**3)
        if torch.distributed.get_rank() == 0:
            total_growth = final_memory - initial_memory
            logger.info(f"update_policy completed - processed {minibatch_count} minibatches")
            logger.info(f"Final GPU memory: {final_memory:.2f}GB (total growth: +{total_growth:.2f}GB)")
        
        return metrics

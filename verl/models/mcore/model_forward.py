# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from verl.utils.megatron_utils import unwrap_model

from .util import (
    postprocess_packed_seqs,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
    postprocess_packed_router_logits,
    recover_left_padding_router_logits,
)


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    use_router_logits=False,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        model_output = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            use_router_logits=use_router_logits,
        )

        # Parse model output consistently across all pipeline stages
        if use_router_logits and isinstance(model_output, tuple) and len(model_output) == 2:
            output_orig, router_logits_raw = model_output
        else:
            output_orig = model_output
            router_logits_raw = None

        # Process primary output (log_probs etc.)
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )

        # Router logits postprocess (packed -> padded) - only if enabled
        if use_router_logits:
            router_logits = postprocess_packed_router_logits(
                router_logits_raw, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
        else:
            router_logits = None
    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        model_output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        
        # Parse model output consistently across all pipeline stages
        if use_router_logits and isinstance(model_output, tuple) and len(model_output) == 2:
            output_orig, router_logits_raw = model_output
        else:
            output_orig = model_output
            router_logits_raw = None

        # recover main output
        output = recover_left_padding(
            output_orig, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
        # router logits recover - only if enabled
        if use_router_logits:
            router_logits = recover_left_padding_router_logits(
                router_logits_raw, new_attention_mask, attention_mask, sequence_length, post_process=post_process
            )
        else:
            router_logits = None
    if value_model and post_process:
        output = output[..., 0]
    # Attach router_logits to output (dict or tensor). Ensure unified return type
    if isinstance(output, dict):
        if use_router_logits:
            output["router_logits"] = router_logits
        # Don't add router_logits key if use_router_logits=False
    else:  # tensor output path (unlikely when logits_processor used)
        if use_router_logits:
            output = {"log_probs": output, "router_logits": router_logits}
        else:
            output = {"log_probs": output}
    
    # if use_router_logits:
    #     print(
    #         f"log_probs shape: {output['log_probs'].shape if 'log_probs' in output else None}, Router logits shape: {output['router_logits'].shape if 'router_logits' in output else None}"
    #     )
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = (
        multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    )
    image_grid_thw = (
        multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    )
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output

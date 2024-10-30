import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, : input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop :, :],
        ],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits
    print("fetched logits")
    targets = input_ids[target_slice]
    print("fetched targets")
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad


def sample_control(
    control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def token_gradients_batch(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    input_ids_adv_portion = torch.gather(input_ids, 1, input_slice).to(
        model.device
    )  # (batch_size, seq_len) -> (batch_size, seq_len_adv)
    # print(input_ids_adv_portion.device)
    one_hot = torch.zeros(
        (
            input_ids_adv_portion.shape[0],
            input_ids_adv_portion.shape[1],
            embed_weights.shape[0],
        ),  # (batch_size, seq_len_adv, vocab_size)
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        -1,
        input_ids_adv_portion.unsqueeze(-1),
        torch.ones(
            one_hot.shape[:-1] + (1,), device=model.device, dtype=embed_weights.dtype
        ),
    )
    one_hot.requires_grad_()
    input_embeds = (
        one_hot @ embed_weights
    )  # (batch_size, seq_len_adv, embed_dim) -> (1, batch_size, seq_len_adv, embed_dim)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids).detach()
    full_embeds = []
    for i in range(embeds.shape[0]):
        # construct the full sequence with the adversarial portion
        # (batch_size, seq_len) -> (batch_size, seq_len_adv) -> (batch_size, seq_len_adv + seq_len_adv + seq_len)
        # print(embeds.shape)
        # print(embeds[i, : input_slice[i], :].shape, input_embeds[i, :, :].shape, embeds[i, input_slice[i] :, :].shape)
        full_embeds.append(
            torch.cat(
                [
                    embeds[i, : input_slice[i, 0], :],
                    input_embeds[i, :, :],
                    embeds[i, input_slice[i, -1] :, :],
                ],
                dim=-2,
            )
        )
    full_embeds = torch.stack(full_embeds)
    # print(full_embeds.shape)

    logits = model(inputs_embeds=full_embeds).logits
    # print("loss slice", loss_slice.shape, loss_slice.device)
    logits_forloss = torch.gather(
        logits, 1, loss_slice.unsqueeze(-1).repeat(1, 1, logits.shape[-1])
    )
    # print(logits_forloss.shape)
    targets = torch.gather(input_ids, 1, target_slice)
    # print(targets.shape)
    loss = nn.CrossEntropyLoss()(
        logits_forloss.view(-1, logits_forloss.shape[-1]), targets.flatten()
    )
    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad, loss


def sample_control(
    control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(
                tokenizer(decoded_str, add_special_tokens=False).input_ids
            ) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(
    *,
    model,
    tokenizer,
    input_ids,
    control_slice,
    test_controls=None,
    return_ids=False,
    batch_size=512,
):

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=model.device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError(
            (
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            )
        )

    locs = (
        torch.arange(control_slice.start, control_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(model.device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids,
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        print(input_ids.shape, ids.shape, attn_mask.shape)
        del locs, test_ids
        gc.collect()
        return (
            forward(
                model=model,
                input_ids=ids,
                attention_mask=attn_mask,
                batch_size=batch_size,
            ),
            ids,
        )
    else:
        del locs, test_ids
        logits = forward(
            model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size
        )
        del ids
        gc.collect()
        return logits


def get_logits_batch(
    *,
    model,
    tokenizer,
    input_ids,
    control_slice,
    test_controls=None,
    return_ids=False,
    batch_size=512,
):

    if isinstance(test_controls[0], str):
        max_len = control_slice.shape[-1]
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=model.device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    # print("nested_ids", nested_ids)
    # print("test_ids", test_ids.shape)
    locs = (
        control_slice.unsqueeze(dim=0).repeat(test_ids.shape[0], 1, 1).to(model.device)
    )  # (num_controls, batch_size, seq_len_adv)
    # print("locs", locs.shape)
    test_ids = test_ids.unsqueeze(dim=1).repeat(1, locs.shape[1], 1)
    # print("test_ids", test_ids.shape)
    # print("input_ids, nochange", input_ids.shape)
    # print("input_ids modified", input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1, 1).shape)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1, 1).to(model.device),
        -1,
        locs,
        test_ids,
    )
    # print(pad_tok)
    # print(tokenizer.pad_token_id)
    # print((ids >= 32000).sum(), "id out of bounds")

    if pad_tok >= 0:
        # print("hi")
        attn_mask = (ids != pad_tok).type(ids.dtype)
        # print("attn_mask", attn_mask.shape)
    else:
        attn_mask = None
    # print(ids.view(-1, ids.shape[-1]).shape)
    # print(attn_mask.view(-1, attn_mask.shape[-1]).shape)
    if return_ids:
        # print(attn_mask)
        del locs, test_ids
        gc.collect()
        return (
            forward(
                model=model,
                input_ids=ids.view(-1, ids.shape[-1]),
                attention_mask=attn_mask.view(-1, attn_mask.shape[-1]),
                batch_size=ids.shape[0],
            ),
            ids,
        )
    else:
        del locs, test_ids
        logits = forward(
            model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size
        )
        del ids
        gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i : i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i : i + batch_size]
        else:
            batch_attention_mask = None
        # print("batch_input_ids", batch_input_ids.shape)
        # print("batch_attention_mask", batch_attention_mask.shape)
        logits.append(
            model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        )
        # print(len(logits))
        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def target_loss_batch(logits, ids, target_slice, og_shape):
    # print("target_slice", target_slice)
    # print("logits", logits.shape)
    # print("ids", ids.shape)
    # print("og_shape", og_shape)
    crit = nn.CrossEntropyLoss(reduction="none")
    # loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    shift_target_slice = target_slice - 1
    # print(ids[:, target_slice])
    targetsf = torch.gather(ids, -1, target_slice.to(ids.device))
    logitsf = torch.gather(
        logits, -2, shift_target_slice.unsqueeze(-1).repeat(1, 1, logits.shape[-1]).to(logits.device)
    )
    print("targets", targetsf)
    print("logits", logitsf.shape)
    loss = crit(
        logitsf.view(-1, logits.shape[-1]),
        targetsf.view(-1),
    )
    # print(loss.shape)
    loss = loss.view(logits.shape[0] // og_shape[0], og_shape[0], -1)
    # print(loss.shape)
    loss = loss.mean(dim=-1).mean(dim=-1)
    # print(loss.shape)
    return loss


def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction="none")
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(
    model_path, tokenizer_path=None, device="cuda:0", **kwargs
):
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs,
            device_map=device,
        ).eval()

    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs
            )
            .to(device)
            .eval()
        )

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    if "oasst-sft-6-llama-30b" in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

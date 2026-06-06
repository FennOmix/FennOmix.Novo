import torch

from .foxnovo import FoxNovoNARModel


def load_encoder_weight(new_model: FoxNovoNARModel, pretrained_path: str) -> None:
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    encoder_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("encoder."):
            encoder_state[key] = value

    missing_keys, unexpected_keys = new_model.load_state_dict(encoder_state, strict=False)

    critical_keys = [key for key in encoder_state if key.startswith("encoder.")]
    missing_critical = [key for key in missing_keys if key in critical_keys]

    if missing_critical:
        raise RuntimeError(f"Critical encoder weights NOT loaded: {missing_critical}")

    if "config" in pretrained_state:
        pretrained_dim = pretrained_state["config"]["dim_model"]
        if pretrained_dim != new_model.encoder.dim_model:
            raise ValueError(
                f"Pretrained model dimension ({pretrained_dim}) "
                f"does not match current model ({new_model.encoder.dim_model})"
            )
    if unexpected_keys:
        print(f"Unexpected keys while loading encoder weights: {unexpected_keys}")
    return new_model


def load_model_weight(new_model: FoxNovoNARModel, pretrained_path: str) -> None:  # noqa: C901
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    state = {}

    for key, value in pretrained_state.items():
        if key.startswith("encoder.diff_encoder.") or key.startswith("encoder.fusion_proj."):
            continue
        # mapping old tokenizer encoder keys to new encoder keys
        if key == "encoder.int_mz_embedding.weight":
            new_key = "encoder.tokenizer_encoder.int_mz_embedding.weight"

        elif key == "encoder.dec_mz_embedding.weight":
            new_key = "encoder.tokenizer_encoder.dec_mz_embedding.weight"

        elif key == "encoder.input_proj.weight":
            new_key = "encoder.tokenizer_encoder.input_proj.weight"

        elif key == "encoder.input_proj.bias":
            new_key = "encoder.tokenizer_encoder.input_proj.bias"

        elif key == "encoder.token_lookup":
            new_key = "encoder.tokenizer_encoder.token_lookup"

        else:
            new_key = key

        if new_key in state:
            raise ValueError(f"Key collision detected: {new_key}")

        state[new_key] = value

    missing_keys, unexpected_keys = new_model.load_state_dict(state, strict=False)

    print(f"Loaded encoder weights from {pretrained_path}")

    critical_keys = [
        "encoder.tokenizer_encoder.int_mz_embedding.weight",
        "encoder.tokenizer_encoder.dec_mz_embedding.weight",
        "encoder.tokenizer_encoder.input_proj.weight",
    ]

    missing_critical = [k for k in missing_keys if k in critical_keys]

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    if missing_critical:
        raise RuntimeError(f"Critical weights NOT loaded: {missing_critical}")

    if "config" in pretrained_state:
        pretrained_dim = pretrained_state["config"]["dim_model"]
        if pretrained_dim != new_model.encoder.dim_model:
            raise ValueError(
                f"Pretrained model dimension ({pretrained_dim}) "
                f"does not match current model ({new_model.encoder.dim_model})"
            )

    return new_model

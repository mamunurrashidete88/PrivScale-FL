from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110, ResNetCIFAR
from .wrn import wrn_28_10, WideResNet
from .femnist_cnn import FemnistCNN
from .char_lstm import CharLSTM
from .gpt2_wrapper import build_gpt2, GPT2LM, GPT2Config

__all__ = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "ResNetCIFAR",
    "wrn_28_10",
    "WideResNet",
    "FemnistCNN",
    "CharLSTM",
    "build_gpt2",
    "GPT2LM",
    "GPT2Config",
]


def build_model(name: str, num_classes: int | None = None, **kwargs):
    """Factory: build a model by name.

    Supported names: resnet20, resnet32, resnet44, resnet56, resnet110,
    wrn_28_10, femnist_cnn, char_lstm, gpt2-small, gpt2-medium, gpt2-large.
    """
    name = name.lower()
    if name == "resnet20":
        return resnet20(num_classes=num_classes or 10)
    if name == "resnet32":
        return resnet32(num_classes=num_classes or 10)
    if name == "resnet44":
        return resnet44(num_classes=num_classes or 10)
    if name == "resnet56":
        return resnet56(num_classes=num_classes or 10)
    if name == "resnet110":
        return resnet110(num_classes=num_classes or 10)
    if name in ("wrn_28_10", "wrn-28-10", "wrn2810"):
        return wrn_28_10(num_classes=num_classes or 100, dropout=kwargs.get("dropout", 0.3))
    if name == "femnist_cnn":
        return FemnistCNN(num_classes=num_classes or 62)
    if name == "char_lstm":
        from .char_lstm import CharLSTM as _CL
        return _CL(
            vocab_size=kwargs.get("vocab_size", 80),
            embed_dim=kwargs.get("embed_dim", 8),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
        )
    if name.startswith("gpt2"):
        return build_gpt2(
            variant=name,
            use_lora=kwargs.get("use_lora", False),
            lora_r=kwargs.get("lora_r", 8),
            lora_alpha=kwargs.get("lora_alpha", 16),
            lora_dropout=kwargs.get("lora_dropout", 0.05),
        )
    raise ValueError(f"Unknown model name: {name}")

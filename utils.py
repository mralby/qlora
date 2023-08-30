import bitsandbytes as bnb
import peft
import json
import shutil
import torch
from peft.utils import _get_submodules
import os


def dequantize_model(model, tokenizer, to='./dequantized_model', dtype=torch.float16):
    """
    After calling this, WE SHOULD LOAD THE MODEL with AutoPeftModel AND CALL merge_and_unload(). NON È VERO -> vedi commento https://github.com/artidoro/qlora/issues/28#issuecomment-1692823931,
        se si elimina la riga commentata, la funzione è equivalente a fare dequantize + merge_and_unload
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    """



    if os.path.exists(to):
        shutil.rmtree(to)

    os.makedirs(to, exist_ok=True)

    cls = peft.tuners.lora.Linear4bit

    base_model = model.base_model.model

    with torch.no_grad():
        for name, module in base_model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                if module.bias is None:
#                    module.disable_adapters = True   # so peft.tuners.lora.Linear4bit.foward is the same as bnb.nn.Linear4bit
                    dequantized_weight = module(torch.eye(module.in_features, dtype=dtype).to(module.weight.device))
                    dequantized_weight = torch.transpose(dequantized_weight, 0, 1).to("cpu")
                    new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None)
                    new_module.weight = torch.nn.Parameter(dequantized_weight)
                else:
                    # TODO: handle when bias is not None
                    raise NotImplementedError

                parent, target, target_name = _get_submodules(base_model, name)
                setattr(parent, target_name, new_module)

        # a hack, setting this to avoid hf's saving error because hf
        # itself does not support saving a model that is registered to be loaded in 4bit.
        base_model.is_loaded_in_4bit = False

        print("Saving dequantized model...")
        base_model.save_pretrained(to)
        tokenizer.save_pretrained(to)
        config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
        config_data.pop("quantization_config", None)
        config_data.pop("pretraining_tp", None)
        with open(os.path.join(to, 'config.json'), 'w') as config:
            config.write(json.dumps(config_data, indent=2))
            
            
def my_dequantize_model(model, tokenizer, to='./dequantized_model', dtype=torch.float16):
    """
    This use bnb.dequantize_4bit. After calling this, WE SHOULD LOAD THE MODEL with AutoPeftModel AND CALL merge_and_unload(). NON È VERO -> vedi commento https://github.com/artidoro/qlora/issues/28#issuecomment-1692823931,
        se si elimina la riga commentata, la funzione è equivalente a fare dequantize + merge_and_unload
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    """

    if os.path.exists(to):
        shutil.rmtree(to)

    os.makedirs(to, exist_ok=True)

    clss = [peft.tuners.lora.Linear4bit, bnb.nn.Linear4bit]

    base_model = model.base_model.model

    with torch.no_grad():
        for name, module in base_model.named_modules():
            if any([isinstance(module, cls) for cls in clss]):
                print(f"Dequantizing `{name}`...")
                if module.bias is None:
#                    module.disable_adapters = True   # so peft.tuners.lora.Linear4bit.foward is the same as bnb.nn.Linear4bit
                    device = module.weight.device
                    module = bnb.dequantize_4bit(module.weight.data, quant_state=module.weight.quant_state, quant_type='nf4').to(device)
                else:
                    # TODO: handle when bias is not None
                    raise NotImplementedError

                parent, target, target_name = _get_submodules(base_model, name)
                setattr(parent, target_name, module)

        # a hack, setting this to avoid hf's saving error because hf
        # itself does not support saving a model that is registered to be loaded in 4bit.
        base_model.is_loaded_in_4bit = False

        print("Saving dequantized model...")
        base_model.save_pretrained(to)
        tokenizer.save_pretrained(to)
        config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
        config_data.pop("quantization_config", None)
        config_data.pop("pretraining_tp", None)
        with open(os.path.join(to, 'config.json'), 'w') as config:
            config.write(json.dumps(config_data, indent=2))
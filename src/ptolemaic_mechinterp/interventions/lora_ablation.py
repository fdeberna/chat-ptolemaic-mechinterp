"""Scoped block-wise removal of unmerged, frozen vanilla LoRA contributions."""
from contextlib import contextmanager


def adapter_state(model, modules):
    """Small state snapshot; never copies weights or activations."""
    return {
        'modules': {
            name: {
                'disabled': bool(module.disable_adapters),
                'active_adapters': list(module.active_adapters),
                'scaling': dict(module.scaling),
                'merged_adapters': list(module.merged_adapters),
                'training': module.training,
            }
            for name, module in modules.items()
        },
        'requires_grad': {name: p.requires_grad for name, p in model.named_parameters()},
    }


def validate_ablation_inventory(model, inventory):
    """Fail closed on a model that differs from the inspected 196-module export."""
    from peft.tuners.lora.bnb import Linear4bit

    expected = {row['expected_peft_module_path']: row for row in inventory}
    if len(expected) != 196 or len(inventory) != 196:
        raise ValueError('Expected exactly 196 unique adapter modules')
    actual = {
        name: module for name, module in model.named_modules()
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')
    }
    if set(actual) != set(expected):
        raise ValueError(f'Runtime adapter paths differ: {set(actual) ^ set(expected)}')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Model must be in evaluation mode with all parameters frozen')
    if set(model.peft_config) != {'default'}:
        raise ValueError('Expected one adapter named default')
    config = model.peft_config['default']
    if (config.use_dora or config.use_rslora or config.bias != 'none'
            or config.lora_bias or config.modules_to_save):
        raise ValueError('Unexpected adapter variant or extra trained parameters')
    for name, module in actual.items():
        row = expected[name]
        if not isinstance(module, Linear4bit):
            raise TypeError(f'{name}: expected quantized PEFT Linear4bit')
        if module.merged or module.merged_adapters:
            raise ValueError(f'{name}: merged adapters are not supported')
        if module.training or module.lora_variant:
            raise ValueError(f'{name}: expected vanilla evaluation-mode LoRA')
        if list(module.active_adapters) != ['default']:
            raise ValueError(f'{name}: unexpected active adapter')
        if set(module.lora_A) != {'default'} or set(module.lora_B) != {'default'}:
            raise ValueError(f'{name}: unexpected factor adapters')
        if module.r['default'] != int(row['rank']) or module.scaling['default'] != float(row['intrinsic_scaling']):
            raise ValueError(f'{name}: rank or intrinsic scaling differs')
        if module.base_layer.in_features != int(row['input_features']) or module.base_layer.out_features != int(row['output_features']):
            raise ValueError(f'{name}: base dimensions differ')
        if tuple(module.lora_A['default'].weight.shape) != (int(row['rank']), int(row['input_features'])):
            raise ValueError(f'{name}: A shape differs')
        if tuple(module.lora_B['default'].weight.shape) != (int(row['output_features']), int(row['rank'])):
            raise ValueError(f'{name}: B shape differs')
    return actual


@contextmanager
def ablate_lora_blocks(model, inventory, saved_layers):
    """Disable all seven adapter branches in selected saved blocks, then restore.

    This changes every token's computation while the scope is active. Begin with
    a fresh KV cache. Does not merge weights, change scaling, or remove blocks.
    """
    layers = tuple(saved_layers)
    if len(set(layers)) != len(layers) or any(type(x) is not int or not 1 <= x <= 28 for x in layers):
        raise ValueError('Saved layers must be unique integers in 1..28')
    modules = validate_ablation_inventory(model, inventory)
    selected = [row['expected_peft_module_path'] for row in inventory
                if int(row['saved_layer']) in layers]
    if len(selected) != 7 * len(layers):
        raise ValueError('Expected exactly seven adapter modules per selected block')
    before = adapter_state(model, modules)
    try:
        for name in selected:
            modules[name].enable_adapters(False)
        yield {'saved_layers': list(layers), 'disabled_modules': selected,
               'module_count': len(selected)}
    finally:
        # Public re-enable may make LoRA factors trainable. No forward occurs
        # during restoration; restore frozen flags immediately afterwards.
        for name in selected:
            module = modules[name]
            state = before['modules'][name]
            module.set_adapter(state['active_adapters'], inference_mode=True)
            module.scaling.clear()
            module.scaling.update(state['scaling'])
            module.enable_adapters(not state['disabled'])
        for name, p in model.named_parameters():
            if p.requires_grad != before['requires_grad'][name]:
                p.requires_grad_(before['requires_grad'][name])
        if adapter_state(model, modules) != before:
            raise RuntimeError('Adapter state was not fully restored')

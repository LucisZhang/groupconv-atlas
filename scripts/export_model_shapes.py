"""Export frozen torchvision convolution shapes from real CPU forward hooks.

Run: python -m scripts.export_model_shapes --output /path/model-shapes.json
Models use weights=None; this captures geometry, not pretrained accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from bench.api import ROOT, Shape

DEFAULT_PROTOCOL = ROOT / 'configs/model-shapes-protocol.json'


def operation_record(shape, bias, padding_mode='zeros'):
    """Preserve original Conv2d semantics separately from no-bias ABI geometry."""
    if not isinstance(bias, bool):
        raise ValueError('bias must be the observed Boolean module property')
    geometry = shape.record()
    attributes = {key: value for key, value in geometry.items() if key != 'shape_id'}
    attributes.update(operator='torch.nn.Conv2d', bias=bias,
                      bias_dims=[shape.cout] if bias else None, padding_mode=padding_mode)
    operation_id = 'conv2d_' + hashlib.sha256(json.dumps(attributes, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()[:16]
    unsupported = bias or padding_mode != 'zeros'
    return {**geometry, 'bias': bias, 'source_has_bias': bias,
            'source_operation_attributes': attributes, 'source_operation_id': operation_id,
            'no_bias_geometry_id': shape.id,
            'shape_id': operation_id if unsupported else shape.id,
            'native_status': 'UNSUPPORTED' if unsupported else 'SUPPORTED',
            'native_reason': 'bias not implemented' if bias else ('padding mode not implemented' if unsupported else None)}


def freeze_record(row):
    keys = ('model', 'module_path', 'bias', 'source_has_bias', 'source_operation_attributes',
            'source_operation_id', 'no_bias_geometry_id', 'shape_id', 'native_status', 'native_reason')
    # Apply source semantics AFTER Shape.record(), which describes only the no-bias ABI.
    return {**Shape.from_record(row).record(), **{key: row[key] for key in keys}}



def validate_protocol(protocol):
    if protocol['torchvision_version'] != '0.23.0':
        raise ValueError('protocol must pin torchvision 0.23.0')
    if protocol['weights'] is not None or protocol['device'] != 'cpu':
        raise ValueError('only CPU weights=None export is authorized')
    if protocol['batches'] != [1, 4] or protocol['input_chw'] != [3, 224, 224]:
        raise ValueError('frozen batch/input dimensions differ')
    models = {'mobilenet_v2', 'resnext50_32x4d', 'convnext_tiny'}
    if set(protocol['selection']) != models or any(len(v) != 2 or len(set(v)) != 2 for v in protocol['selection'].values()):
        raise ValueError('require two unique selected modules for each model')
    if protocol['expected_record_count'] != 12:
        raise ValueError('expected exactly 12 model/batch shapes')


def export_shapes(protocol_path=DEFAULT_PROTOCOL):
    protocol_path = Path(protocol_path)
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    validate_protocol(protocol)
    import torch
    import torchvision
    version = str(torchvision.__version__)
    if version.split('+')[0] != protocol['torchvision_version']:
        raise RuntimeError(f'torchvision version mismatch: expected 0.23.0, observed {version}')
    rows, inventories = [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for model_name, selected in protocol['selection'].items():
            # Restore process RNG afterward; fresh model init has a fixed identity.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(protocol['seed'])
                model = getattr(torchvision.models, model_name)(weights=None).cpu().float().eval()
            modules = dict(model.named_modules())
            if any(name not in modules or not isinstance(modules[name], torch.nn.Conv2d) for name in selected):
                raise ValueError(f'selected Conv2d module not found: {model_name}')
            state_hash = hashlib.sha256()
            for name, value in model.state_dict().items():
                state_hash.update(name.encode()); state_hash.update(value.detach().cpu().numpy().tobytes())
            for batch in protocol['batches']:
                observed = []
                handles = []
                def hook_for(name):
                    def hook(module, inputs, output):
                        x = inputs[0]
                        shape = Shape(n=int(x.shape[0]), cin=module.in_channels, cout=module.out_channels,
                            h=int(x.shape[2]), w=int(x.shape[3]), r=module.kernel_size[0], s=module.kernel_size[1],
                            groups=module.groups, sh=module.stride[0], sw=module.stride[1],
                            ph=module.padding[0], pw=module.padding[1], dh=module.dilation[0], dw=module.dilation[1])
                        if tuple(output.shape) != shape.output:
                            raise ValueError('observed output disagrees with shape formula')
                        observed.append((name, shape, module.bias is not None, module.padding_mode))
                    return hook
                # Hook every Conv2d, and retain inventory as a closed forward trace.
                for name, module in modules.items():
                    if isinstance(module, torch.nn.Conv2d):
                        handles.append(module.register_forward_hook(hook_for(name)))
                try:
                    generator = torch.Generator(device='cpu').manual_seed(protocol['seed'])
                    x = torch.randn((batch, *protocol['input_chw']), generator=generator, dtype=torch.float32)
                    with torch.inference_mode():
                        model(x)
                finally:
                    for handle in handles:
                        handle.remove()
                inventories.append({'model': model_name, 'batch': batch,
                                    'conv_invocations': len(observed), 'module_paths': [name for name, _, _, _ in observed]})
                for name in selected:
                    matches = [(shape, bias, padding_mode) for path, shape, bias, padding_mode in observed if path == name]
                    if len(matches) != 1:
                        raise ValueError(f'expected exactly one invocation for {model_name}/{name}')
                    shape, bias, padding_mode = matches[0]
                    rows.append({**operation_record(shape, bias, padding_mode), 'model': model_name, 'module_path': name,
                        'torchvision_version': version, 'torch_version': str(torch.__version__),
                        'seed': protocol['seed'], 'weights': None, 'weights_identity': protocol['weights_identity'],
                        'initialized_state_sha256': state_hash.hexdigest(), 'shape_source': 'actual_cpu_forward_hook'})
            del model
    finally:
        torch.set_num_threads(previous_threads)
    if len(rows) != protocol['expected_record_count'] or len({row['shape_id'] for row in rows}) != len(rows):
        raise ValueError('record count or shape uniqueness mismatch')
    if 'frozen_shapes' in protocol:
        observed = [freeze_record(row) for row in rows]
        if json.loads(json.dumps(observed)) != protocol['frozen_shapes']:
            raise ValueError('actual forward geometry differs from frozen shape protocol')
    for status, key in (('SUPPORTED', 'expected_native_supported_count'), ('UNSUPPORTED', 'expected_native_unsupported_count')):
        if key in protocol and sum(row['native_status'] == status for row in rows) != protocol[key]:
            raise ValueError('native support denominator differs from frozen protocol')
    if not any(row['sh'] == 2 or row['sw'] == 2 for row in rows):
        raise ValueError('missing stride-2 coverage')
    return {'schema_version': 2, 'kind': 'model_shape_export', 'protocol_id': protocol['protocol_id'],
            'protocol_sha256': hashlib.sha256(raw).hexdigest(), 'torchvision_version': version,
            'torch_version': str(torch.__version__), 'device': 'cpu', 'seed': protocol['seed'],
            'weights': None, 'scope': protocol['scope'], 'forward_inventory': inventories, 'shapes': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_shapes(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"exported {len(report['shapes'])} actual forward shapes to {args.output}")


if __name__ == '__main__':
    main()

"""Small CPU validation for candidate construction and zero-init behavior."""

import torch

from model.smile import SMILE2, cassi_measure, phi_phi_t, shift_cube


def main():
    torch.manual_seed(7)
    common = dict(
        dim=28, unet_stage=2, num_blocks=[1, 1, 1], use_spatial_content_modulation=True,
        use_perchannel=True, use_spectral_wave=True, post_block="ffn", ffn_mult=2,
        input_mode="H", output_dc=False, swp_variant="full", bands=28,
    )
    base = SMILE2(**common)
    state = base.state_dict()
    batch, height, width = 1, 16, 16
    mask = torch.rand(batch, 28, height, width)
    gt = torch.rand(batch, 28, height, width)
    y = cassi_measure(gt, mask)
    shifted = shift_cube(mask)
    ppt = phi_phi_t(mask)
    with torch.no_grad():
        reference = base(y, mask, shifted, ppt)

    for adapter in ("mask", "wavelength"):
        model = SMILE2(**common, input_adapter=adapter)
        missing, unexpected = model.load_state_dict(state, strict=False)
        with torch.no_grad():
            output = model(y, mask, shifted, ppt)
        difference = float((output - reference).abs().max())
        print(
            "%s finite=%s max_base_diff=%.9g new_keys=%d unexpected=%d"
            % (adapter, bool(torch.isfinite(output).all()), difference, len(missing), len(unexpected))
        )
        if difference != 0.0 or unexpected:
            raise SystemExit("zero-init equivalence failed for %s" % adapter)

    model = SMILE2(**common, wave_param_mode="symmetric_basis", wave_basis_count=3)
    output = model(y, mask, shifted, ppt)
    loss = (output - gt).square().mean()
    loss.backward()
    finite_grad = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    print("ms_mode finite=%s finite_grad=%s loss=%.6f" % (
        bool(torch.isfinite(output).all()), finite_grad, float(loss)
    ))
    if not finite_grad:
        raise SystemExit("symmetric basis produced non-finite gradients")
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SpectralWavePropagator":
            alpha, vs, _, _ = module._get_effective_params()
            indices = (-torch.arange(alpha.numel())) % alpha.numel()
            alpha_error = float((alpha.flatten() - alpha.flatten()[indices]).abs().max())
            vs_error = float((vs.flatten() - vs.flatten()[indices]).abs().max())
            print("symmetry %s alpha=%.9g vs=%.9g" % (name, alpha_error, vs_error))
            if alpha_error > 1e-7 or vs_error > 1e-7:
                raise SystemExit("frequency symmetry validation failed")
            break


if __name__ == "__main__":
    main()




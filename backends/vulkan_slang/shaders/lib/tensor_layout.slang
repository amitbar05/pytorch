module tensor_layout;

[ForceInline]
public uint contiguous_offset<let N : int>(uint linear, uint shape[N]) {
    return linear;
}

[ForceInline]
public uint strided_offset<let N : int>(
    uint linear, uint shape[N], uint stride[N]
) {
    uint offset = 0u;
    [unroll] for (int i = N - 1; i >= 0; i--) {
        uint coord = linear % shape[i];
        linear /= shape[i];
        offset += coord * stride[i];
    }
    return offset;
}

[ForceInline]
public uint broadcast_offset<let N : int>(
    uint out_linear, uint out_shape[N], uint src_shape[N], uint src_stride[N]
) {
    uint offset = 0u;
    [unroll] for (int i = N - 1; i >= 0; i--) {
        uint coord = out_linear % out_shape[i];
        out_linear /= out_shape[i];
        if (src_shape[i] != 1u) offset += coord * src_stride[i];
    }
    return offset;
}

public void unravel_index<let N : int>(
    uint linear, uint shape[N], out uint coords[N]
) {
    [unroll] for (int i = N - 1; i >= 0; i--) {
        coords[i] = linear % shape[i];
        linear /= shape[i];
    }
}

[ForceInline]
public uint ravel_index<let N : int>(uint coords[N], uint stride[N]) {
    uint offset = 0u;
    [unroll] for (int i = 0; i < N; i++) {
        offset += coords[i] * stride[i];
    }
    return offset;
}

[ForceInline]
public uint numel<let N : int>(uint shape[N]) {
    uint n = 1u;
    [unroll] for (int i = 0; i < N; i++) {
        n *= shape[i];
    }
    return n;
}

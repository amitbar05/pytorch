module mm;

[[vk::constant_id(0)]] public const uint TILE_M = 16;
[[vk::constant_id(1)]] public const uint TILE_N = 16;
[[vk::constant_id(2)]] public const uint TILE_K = 16;

public interface IEpilogue {
    static float apply(float acc, uint m, uint n);
};

public struct EpilogueIdentity : IEpilogue {
    static float apply(float acc, uint m, uint n) { return acc; }
};

public struct EpilogueReLU : IEpilogue {
    static float apply(float acc, uint m, uint n) { return max(acc, 0.0f); }
};

public struct EpilogueClampPositive : IEpilogue {
    static float apply(float acc, uint m, uint n) { return clamp(acc, 0.0f, 6.0f); }
};

groupshared float _mm_smem_a[1024];
groupshared float _mm_smem_b[1024];

public void mm_tiled<Epi : IEpilogue>(
    StructuredBuffer<float> A,
    StructuredBuffer<float> B,
    RWStructuredBuffer<float> C,
    uint M, uint N, uint K,
    uint3 gid, uint3 lid
) {
    uint row = gid.y * TILE_M + lid.y;
    uint col = gid.x * TILE_N + lid.x;
    float acc = 0.0f;

    for (uint kk = 0u; kk < K; kk += TILE_K) {
        uint a_off = lid.y * TILE_K + lid.x;
        if (lid.x < TILE_K) {
            uint a_row = gid.y * TILE_M + lid.y;
            uint a_col = kk + lid.x;
            _mm_smem_a[a_off] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        uint b_off = lid.y * TILE_N + lid.x;
        if (lid.y < TILE_K) {
            uint b_row = kk + lid.y;
            uint b_col = gid.x * TILE_N + lid.x;
            _mm_smem_b[b_off] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint kt = 0u; kt < TILE_K; kt++) {
            acc += _mm_smem_a[lid.y * TILE_K + kt] * _mm_smem_b[kt * TILE_N + lid.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N) {
        C[row * N + col] = Epi::apply(acc, row, col);
    }
}

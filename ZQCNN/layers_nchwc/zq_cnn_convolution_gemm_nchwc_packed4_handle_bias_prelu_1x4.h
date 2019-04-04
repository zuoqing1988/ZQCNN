#if WITH_BIAS
bias_v = zq_mm_load_ps(bias + out_c);
a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
slope_v = zq_mm_load_ps(slope + out_c);
c00 = zq_mm_min_ps(a0, zero_v);
c10 = zq_mm_max_ps(a0, zero_v);
a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
slope_v = zq_mm_load_ps(slope + out_c);
a0 = zq_mm_load_ps(dst_ptr0);
c00 = zq_mm_min_ps(a0, zero_v);
c10 = zq_mm_max_ps(a0, zero_v);
zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
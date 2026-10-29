#!/usr/bin/env python3
"""Generates src/web/include/rdna_ops.h: gfx10 opcode numbers, taken from LLVM's assembler.

usage: python3 tools/rdna/gen_ops.py [llvm-mc]   (needs an llvm-mc with the AMDGPU target)
"""
import os, re, subprocess, sys

LLVM_MC = sys.argv[1] if len(sys.argv) > 1 else '/opt/homebrew/opt/llvm/bin/llvm-mc'

SOP2 = '''s_add_u32 s_sub_u32 s_add_i32 s_sub_i32 s_addc_u32 s_subb_u32 s_min_i32 s_min_u32 s_max_i32 s_max_u32
s_cselect_b32 s_cselect_b64 s_and_b32 s_and_b64 s_or_b32 s_or_b64 s_xor_b32 s_xor_b64 s_andn2_b32 s_andn2_b64
s_orn2_b32 s_orn2_b64 s_nand_b32 s_nand_b64 s_nor_b32 s_nor_b64 s_xnor_b32 s_xnor_b64 s_lshl_b32 s_lshl_b64
s_lshr_b32 s_lshr_b64 s_ashr_i32 s_ashr_i64 s_bfm_b32 s_bfm_b64 s_mul_i32 s_bfe_u32 s_bfe_i32 s_bfe_u64 s_bfe_i64
s_absdiff_i32 s_lshl1_add_u32 s_lshl2_add_u32 s_lshl3_add_u32 s_lshl4_add_u32 s_pack_ll_b32_b16
s_pack_lh_b32_b16 s_pack_hh_b32_b16 s_mul_hi_u32 s_mul_hi_i32'''
SOPK = '''s_movk_i32 s_cmovk_i32 s_cmpk_eq_i32 s_cmpk_lg_i32 s_cmpk_gt_i32 s_cmpk_ge_i32 s_cmpk_lt_i32 s_cmpk_le_i32
s_cmpk_eq_u32 s_cmpk_lg_u32 s_cmpk_gt_u32 s_cmpk_ge_u32 s_cmpk_lt_u32 s_cmpk_le_u32 s_addk_i32 s_mulk_i32
s_getreg_b32 s_setreg_b32 s_setreg_imm32_b32 s_call_b64 s_waitcnt_vscnt s_waitcnt_vmcnt s_waitcnt_expcnt
s_waitcnt_lgkmcnt s_subvector_loop_begin s_subvector_loop_end s_version'''
SOP1 = '''s_mov_b32 s_mov_b64 s_cmov_b32 s_cmov_b64 s_not_b32 s_not_b64 s_wqm_b32 s_wqm_b64 s_brev_b32 s_brev_b64
s_bcnt0_i32_b32 s_bcnt0_i32_b64 s_bcnt1_i32_b32 s_bcnt1_i32_b64 s_ff0_i32_b32 s_ff0_i32_b64 s_ff1_i32_b32
s_ff1_i32_b64 s_flbit_i32_b32 s_flbit_i32_b64 s_flbit_i32 s_flbit_i32_i64 s_sext_i32_i8 s_sext_i32_i16
s_bitset0_b32 s_bitset0_b64 s_bitset1_b32 s_bitset1_b64 s_getpc_b64 s_setpc_b64 s_swappc_b64 s_rfe_b64
s_and_saveexec_b64 s_or_saveexec_b64 s_xor_saveexec_b64 s_andn2_saveexec_b64 s_orn2_saveexec_b64
s_nand_saveexec_b64 s_nor_saveexec_b64 s_xnor_saveexec_b64 s_quadmask_b32 s_quadmask_b64 s_movrels_b32
s_movrels_b64 s_movreld_b32 s_movreld_b64 s_abs_i32 s_andn1_saveexec_b64 s_orn1_saveexec_b64
s_andn1_wrexec_b64 s_andn2_wrexec_b64 s_bitreplicate_b64_b32 s_and_saveexec_b32 s_or_saveexec_b32
s_xor_saveexec_b32 s_andn2_saveexec_b32 s_orn2_saveexec_b32 s_nand_saveexec_b32 s_nor_saveexec_b32
s_xnor_saveexec_b32 s_andn1_saveexec_b32 s_orn1_saveexec_b32 s_andn1_wrexec_b32 s_andn2_wrexec_b32
s_movrelsd_2_b32'''
SOPC = '''s_cmp_eq_i32 s_cmp_lg_i32 s_cmp_gt_i32 s_cmp_ge_i32 s_cmp_lt_i32 s_cmp_le_i32 s_cmp_eq_u32 s_cmp_lg_u32
s_cmp_gt_u32 s_cmp_ge_u32 s_cmp_lt_u32 s_cmp_le_u32 s_bitcmp0_b32 s_bitcmp1_b32 s_bitcmp0_b64 s_bitcmp1_b64
s_cmp_eq_u64 s_cmp_lg_u64'''
SOPP = '''s_nop s_endpgm s_branch s_wakeup s_cbranch_scc0 s_cbranch_scc1 s_cbranch_vccz s_cbranch_vccnz
s_cbranch_execz s_cbranch_execnz s_barrier s_setkill s_waitcnt s_sethalt s_sleep s_setprio s_sendmsg
s_sendmsghalt s_trap s_icache_inv s_incperflevel s_decperflevel s_ttracedata s_cbranch_cdbgsys
s_cbranch_cdbguser s_cbranch_cdbgsys_or_user s_cbranch_cdbgsys_and_user s_endpgm_saved s_endpgm_ordered_ps_done
s_code_end s_inst_prefetch s_clause s_round_mode s_denorm_mode s_ttracedata_imm'''
SMEM = '''s_load_dword s_load_dwordx2 s_load_dwordx4 s_load_dwordx8 s_load_dwordx16 s_buffer_load_dword
s_buffer_load_dwordx2 s_buffer_load_dwordx4 s_buffer_load_dwordx8 s_buffer_load_dwordx16 s_gl1_inv s_dcache_inv
s_memtime s_memrealtime'''
VOP2 = '''v_cndmask_b32 v_add_f32 v_sub_f32 v_subrev_f32 v_mac_legacy_f32 v_mul_legacy_f32 v_mul_f32 v_mul_i32_i24
v_mul_hi_i32_i24 v_mul_u32_u24 v_mul_hi_u32_u24 v_min_f32 v_max_f32 v_min_i32 v_max_i32 v_min_u32 v_max_u32
v_lshrrev_b32 v_ashrrev_i32 v_lshlrev_b32 v_and_b32 v_or_b32 v_xor_b32 v_xnor_b32 v_mac_f32 v_madmk_f32
v_madak_f32 v_add_nc_u32 v_sub_nc_u32 v_subrev_nc_u32 v_add_co_ci_u32 v_sub_co_ci_u32 v_subrev_co_ci_u32
v_fmac_f32 v_fmamk_f32 v_fmaak_f32 v_cvt_pkrtz_f16_f32 v_add_f16 v_sub_f16 v_subrev_f16 v_mul_f16 v_fmac_f16
v_fmamk_f16 v_fmaak_f16 v_max_f16 v_min_f16 v_ldexp_f16 v_pk_fmac_f16'''
VOP1 = '''v_nop v_mov_b32 v_readfirstlane_b32 v_cvt_i32_f64 v_cvt_f64_i32 v_cvt_f32_i32 v_cvt_f32_u32 v_cvt_u32_f32
v_cvt_i32_f32 v_cvt_f16_f32 v_cvt_f32_f16 v_cvt_rpi_i32_f32 v_cvt_flr_i32_f32 v_cvt_off_f32_i4 v_cvt_f32_f64
v_cvt_f64_f32 v_cvt_f32_ubyte0 v_cvt_f32_ubyte1 v_cvt_f32_ubyte2 v_cvt_f32_ubyte3 v_cvt_u32_f64 v_cvt_f64_u32
v_trunc_f64 v_ceil_f64 v_rndne_f64 v_floor_f64 v_pipeflush v_fract_f32 v_trunc_f32 v_ceil_f32 v_rndne_f32
v_floor_f32 v_exp_f32 v_log_f32 v_rcp_f32 v_rcp_iflag_f32 v_rsq_f32 v_rcp_f64 v_rsq_f64 v_sqrt_f32 v_sqrt_f64
v_sin_f32 v_cos_f32 v_not_b32 v_bfrev_b32 v_ffbh_u32 v_ffbl_b32 v_ffbh_i32 v_frexp_exp_i32_f64
v_frexp_mant_f64 v_fract_f64 v_frexp_exp_i32_f32 v_frexp_mant_f32 v_clrexcp v_movreld_b32 v_movrels_b32
v_movrelsd_b32 v_movrelsd_2_b32 v_cvt_f16_u16 v_cvt_f16_i16 v_cvt_u16_f16 v_cvt_i16_f16 v_rcp_f16 v_sqrt_f16
v_rsq_f16 v_log_f16 v_exp_f16 v_floor_f16 v_ceil_f16 v_trunc_f16 v_rndne_f16 v_fract_f16 v_sin_f16 v_cos_f16
v_cvt_norm_i16_f16 v_cvt_norm_u16_f16 v_sat_pk_u8_i16 v_swap_b32'''
VOPC_F = 'f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru'.split()
VOPC_I = 'f lt eq le gt ne ge t'.split()
VOPC = ' '.join(['v_cmp_%s_f32' % c for c in VOPC_F] + ['v_cmpx_%s_f32' % c for c in VOPC_F] +
                ['v_cmp_%s_f64' % c for c in VOPC_F] + ['v_cmpx_%s_f64' % c for c in VOPC_F] +
                ['v_cmp_%s_f16' % c for c in VOPC_F] + ['v_cmpx_%s_f16' % c for c in VOPC_F] +
                ['v_cmp_%s_%s' % (c, t) for t in ('i32', 'u32', 'i64', 'u64') for c in VOPC_I] +
                ['v_cmpx_%s_%s' % (c, t) for t in ('i32', 'u32', 'i64', 'u64') for c in VOPC_I] +
                ['v_cmp_%s_%s' % (c, t) for t in ('i16', 'u16') for c in 'lt eq le gt ne ge'.split()] +
                ['v_cmp_class_f32', 'v_cmpx_class_f32', 'v_cmp_class_f64', 'v_cmpx_class_f64',
                 'v_cmp_class_f16', 'v_cmpx_class_f16'])
VOP3 = '''v_mad_legacy_f32 v_mad_f32 v_mad_i32_i24 v_mad_u32_u24 v_cubeid_f32 v_cubesc_f32 v_cubetc_f32 v_cubema_f32
v_bfe_u32 v_bfe_i32 v_bfi_b32 v_fma_f32 v_fma_f64 v_lerp_u8 v_alignbit_b32 v_alignbyte_b32 v_mullit_f32
v_min3_f32 v_min3_i32 v_min3_u32 v_max3_f32 v_max3_i32 v_max3_u32 v_med3_f32 v_med3_i32 v_med3_u32 v_sad_u8
v_sad_hi_u8 v_sad_u16 v_sad_u32 v_cvt_pk_u8_f32 v_div_fixup_f32 v_div_fixup_f64 v_add_f64 v_mul_f64 v_min_f64
v_max_f64 v_ldexp_f64 v_mul_lo_u32 v_mul_hi_u32 v_mul_hi_i32 v_div_scale_f32 v_div_scale_f64 v_div_fmas_f32
v_div_fmas_f64 v_msad_u8 v_qsad_pk_u16_u8 v_mqsad_pk_u16_u8 v_trig_preop_f64 v_mqsad_u32_u8 v_mad_u64_u32
v_mad_i64_i32 v_xor3_b32 v_lshlrev_b64 v_lshrrev_b64 v_ashrrev_i64 v_add_co_u32 v_sub_co_u32 v_subrev_co_u32
v_add_nc_u16 v_sub_nc_u16 v_mul_lo_u16 v_lshrrev_b16 v_ashrrev_i16 v_max_u16 v_max_i16 v_min_u16 v_min_i16
v_add_nc_i16 v_sub_nc_i16 v_pack_b32_f16 v_cvt_pknorm_i16_f16 v_cvt_pknorm_u16_f16 v_lshlrev_b16
v_mad_u16 v_mad_i16 v_fma_f16 v_div_fixup_f16 v_interp_p1ll_f16 v_interp_p1lv_f16 v_interp_p2_f16
v_add_nc_i32 v_sub_nc_i32 v_lshl_add_u32 v_add_lshl_u32 v_add3_u32 v_lshl_or_b32 v_and_or_b32 v_or3_b32
v_mad_u32_u16 v_mad_i32_i16 v_permlane16_b32 v_permlanex16_b32 v_cvt_pknorm_i16_f32 v_cvt_pknorm_u16_f32
v_cvt_pk_u16_u32 v_cvt_pk_i16_i32 v_readlane_b32 v_writelane_b32 v_ldexp_f32 v_bfm_b32 v_bcnt_u32_b32
v_mbcnt_lo_u32_b32 v_mbcnt_hi_u32_b32 v_cvt_f32_f16 v_interp_p1_f32 v_interp_p2_f32
v_interp_mov_f32'''
VOP3P = '''v_pk_mad_i16 v_pk_mul_lo_u16 v_pk_add_i16 v_pk_sub_i16 v_pk_lshlrev_b16 v_pk_lshrrev_b16 v_pk_ashrrev_i16
v_pk_max_i16 v_pk_min_i16 v_pk_mad_u16 v_pk_add_u16 v_pk_sub_u16 v_pk_max_u16 v_pk_min_u16 v_pk_fma_f16
v_pk_add_f16 v_pk_mul_f16 v_pk_min_f16 v_pk_max_f16 v_fma_mix_f32 v_fma_mixlo_f16 v_fma_mixhi_f16'''
VINTRP = 'v_interp_p1_f32 v_interp_p2_f32 v_interp_mov_f32'
DS = '''ds_add_u32 ds_sub_u32 ds_rsub_u32 ds_inc_u32 ds_dec_u32 ds_min_i32 ds_max_i32 ds_min_u32 ds_max_u32
ds_and_b32 ds_or_b32 ds_xor_b32 ds_mskor_b32 ds_write_b32 ds_write2_b32 ds_write2st64_b32 ds_cmpst_b32
ds_min_f32 ds_max_f32 ds_add_f32 ds_write_b8 ds_write_b16 ds_add_rtn_u32 ds_sub_rtn_u32 ds_min_rtn_i32
ds_max_rtn_i32 ds_min_rtn_u32 ds_max_rtn_u32 ds_and_rtn_b32 ds_or_rtn_b32 ds_xor_rtn_b32 ds_wrxchg_rtn_b32
ds_cmpst_rtn_b32 ds_add_rtn_f32 ds_swizzle_b32 ds_read_b32 ds_read2_b32 ds_read2st64_b32 ds_read_i8 ds_read_u8
ds_read_i16 ds_read_u16 ds_consume ds_append ds_write_b64 ds_write2_b64 ds_write2st64_b64 ds_read_b64
ds_read2_b64 ds_read2st64_b64 ds_permute_b32 ds_bpermute_b32 ds_write_b96 ds_write_b128 ds_read_b96
ds_read_b128 ds_read_u8_d16 ds_read_u16_d16 ds_read_u8_d16_hi ds_read_u16_d16_hi ds_write_b8_d16_hi
ds_write_b16_d16_hi'''
MUBUF = '''buffer_load_format_x buffer_load_format_xy buffer_load_format_xyz buffer_load_format_xyzw
buffer_store_format_x buffer_store_format_xy buffer_store_format_xyz buffer_store_format_xyzw buffer_load_ubyte
buffer_load_sbyte buffer_load_ushort buffer_load_sshort buffer_load_dword buffer_load_dwordx2
buffer_load_dwordx4 buffer_load_dwordx3 buffer_store_byte buffer_store_short buffer_store_dword
buffer_store_dwordx2 buffer_store_dwordx4 buffer_store_dwordx3 buffer_atomic_swap buffer_atomic_cmpswap
buffer_atomic_add buffer_atomic_sub buffer_atomic_smin buffer_atomic_umin buffer_atomic_smax buffer_atomic_umax
buffer_atomic_and buffer_atomic_or buffer_atomic_xor buffer_atomic_inc buffer_atomic_dec buffer_gl0_inv
buffer_gl1_inv buffer_load_format_d16_x buffer_load_format_d16_xy buffer_load_format_d16_xyz
buffer_load_format_d16_xyzw buffer_store_format_d16_x buffer_store_format_d16_xy buffer_store_format_d16_xyz
buffer_store_format_d16_xyzw'''
MTBUF = '''tbuffer_load_format_x tbuffer_load_format_xy tbuffer_load_format_xyz tbuffer_load_format_xyzw
tbuffer_store_format_x tbuffer_store_format_xy tbuffer_store_format_xyz tbuffer_store_format_xyzw'''
MIMG = '''image_load image_load_mip image_load_pck image_load_pck_sgn image_load_mip_pck image_load_mip_pck_sgn
image_store image_store_mip image_store_pck image_store_mip_pck image_get_resinfo image_atomic_swap
image_atomic_cmpswap image_atomic_add image_atomic_sub image_atomic_smin image_atomic_umin image_atomic_smax
image_atomic_umax image_atomic_and image_atomic_or image_atomic_xor image_atomic_inc image_atomic_dec
image_sample image_sample_cl image_sample_d image_sample_d_cl image_sample_l image_sample_b image_sample_b_cl
image_sample_lz image_sample_c image_sample_c_cl image_sample_c_d image_sample_c_d_cl image_sample_c_l
image_sample_c_b image_sample_c_b_cl image_sample_c_lz image_sample_o image_sample_cl_o image_sample_d_o
image_sample_d_cl_o image_sample_l_o image_sample_b_o image_sample_b_cl_o image_sample_lz_o image_sample_c_o
image_sample_c_cl_o image_sample_c_d_o image_sample_c_d_cl_o image_sample_c_l_o image_sample_c_b_o
image_sample_c_b_cl_o image_sample_c_lz_o image_gather4 image_gather4_cl image_gather4_l image_gather4_b
image_gather4_b_cl image_gather4_lz image_gather4_c image_gather4_c_cl image_gather4_c_l image_gather4_c_b
image_gather4_c_b_cl image_gather4_c_lz image_gather4_o image_gather4_lz_o image_gather4_c_lz_o image_get_lod
image_sample_cd image_sample_cd_cl image_sample_c_cd image_sample_c_cd_cl'''
FLAT = '''global_load_ubyte global_load_sbyte global_load_ushort global_load_sshort global_load_dword
global_load_dwordx2 global_load_dwordx4 global_load_dwordx3 global_store_byte global_store_short
global_store_dword global_store_dwordx2 global_store_dwordx4 global_store_dwordx3 global_atomic_swap
global_atomic_cmpswap global_atomic_add global_atomic_sub global_atomic_smin global_atomic_umin global_atomic_smax
global_atomic_umax global_atomic_and global_atomic_or global_atomic_xor global_atomic_inc global_atomic_dec
'''

def candidates(fmt, m):
    s32, s64 = 's0', 's[0:1]'
    if fmt == 'SOP2':
        return ['s0, s1, s2', 's[0:1], s[2:3], s[4:5]', 's[0:1], s[2:3], s4', 's[0:1], s2, s3']
    if fmt == 'SOPK':
        return ['s0, 0x1', 's[0:1], 0x1', 'hwreg(HW_REG_MODE), s0', 's0, hwreg(HW_REG_MODE)',
                'hwreg(HW_REG_MODE), 0x1', 'null, 0x1', '0x1', 's0, 0x1, 0x1']
    if fmt == 'SOP1':
        return ['s0, s1', 's[0:1], s[2:3]', 's0, s[2:3]', 's[0:1], s2', 's[0:1]', 's0']
    if fmt == 'SOPC':
        return ['s0, s1', 's[0:1], s[2:3]', 's[0:1], s2']
    if fmt == 'SOPP':
        return ['', '0', '1', 'sendmsg(MSG_GS_ALLOC_REQ)']
    if fmt == 'SMEM':
        return ['s0, s[2:3], 0x0', 's[0:1], s[2:3], 0x0', 's[0:3], s[2:3], 0x0', 's[0:7], s[2:3], 0x0',
                's[0:15], s[2:3], 0x0', 's[0:3], s[4:7], 0x0', 's[0:7], s[8:11], 0x0',
                's[0:15], s[16:19], 0x0', 's0, s[4:7], 0x0', 's[0:1], s[4:7], 0x0', 's[0:3], s[4:7], 0x0',
                's[0:7], s[8:11], 0x0', 's[0:15], s[16:19], 0x0', '', 's[0:1]']
    if fmt == 'VOP2':
        return ['v0, v1, v2', 'v0, vcc_lo, v1, v2, vcc_lo', 'v0, vcc, v1, v2, vcc', 'v0, v1, v2, vcc',
                'v0, v1, 0x1, v2', 'v0, v1, v2, 0x1']
    if fmt == 'VOP1':
        return ['v0, v1', 'v[0:1], v[2:3]', 'v0, v[2:3]', 'v[0:1], v2', 's0, v1', '']
    if fmt == 'VOPC':
        return ['vcc, v1, v2', 'vcc, v[0:1], v[2:3]', 'v1, v2', 'v[0:1], v[2:3]', 'vcc_lo, v1, v2',
                'vcc_lo, v[0:1], v[2:3]', 'vcc, v[0:1], v2', 'v[0:1], v2']
    if fmt == 'VOP3':
        return ['v0, v1, v2, v3', 'v0, v1, v2', 'v[0:1], v[2:3], v[4:5], v[6:7]', 'v[0:1], v[2:3], v[4:5]',
                'v[0:1], v2, v[4:5]', 'v[0:1], v2, v3', 'v0, vcc, v1, v2', 'v[0:1], vcc, v[2:3], v[4:5], v[6:7]',
                'v0, vcc, v1, v2, v3', 'v[0:1], s[2:3], v1, v2, v[4:5]', 'v0, s[2:3], v1, v2',
                'v[0:1], v2, v[4:5], v[6:7]', 's0, v1, s2', 'v0, s1, s2', 'v0, v1, s2, s3', 'v[0:1], v[2:3], v4',
                'v0, v1, attr0.x', 'v0, v1, attr0.x, v2', 'v0, p10, attr0.x', 'v0, v[2:3], v4',
                'v[0:1], v[2:3], v[4:5], v[6:7]', 'v[0:1], v2, v[4:5], v[6:7]', 'v0, v[2:3], v4, v5',
                'v[0:1], v[2:3], v4, v[6:7]', 'v[0:3], v[4:5], v6, v[8:11]', 'v0, v1',
                'v0, vcc, v1, v2, v3', 'v[0:1], vcc, v[2:3], v[4:5], v[6:7]', 'v[0:1], vcc, v2, v3, v[4:5]']
    if fmt == 'VOP3P':
        return ['v0, v1, v2, v3', 'v0, v1, v2']
    if fmt == 'VINTRP':
        return ['v0, v1, attr0.x', 'v0, p10, attr0.x']
    if fmt == 'DS':
        return ['v0, v1', 'v0, v1, v2', 'v0, v1 offset1:1', 'v0, v1, v2 offset1:1', 'v0, v1, v2, v3',
                'v0, v1, v2, v3 offset1:1', 'v0, v[2:3]', 'v[0:1], v2', 'v0, v[2:3], v[4:5] offset1:1',
                'v[0:3], v4 offset1:1', 'v[0:3], v4', 'v0, v[2:4]', 'v0, v[2:5]', 'v[0:2], v4', 'v0',
                'v0, v1 offset:0', 'v0, v1, v2 offset:0']
    if fmt == 'MUBUF':
        return ['v0, v1, s[0:3], 0 idxen', 'v[0:1], v2, s[0:3], 0 idxen', 'v[0:2], v4, s[0:3], 0 idxen',
                'v[0:3], v4, s[0:3], 0 idxen', '', 'v[0:1], v2, s[0:3], 0 idxen glc']
    if fmt == 'MTBUF':
        f = ' format:[BUF_FMT_32_FLOAT] idxen'
        return ['v0, v1, s[0:3], 0' + f, 'v[0:1], v2, s[0:3], 0' + f, 'v[0:2], v4, s[0:3], 0' + f,
                'v[0:3], v4, s[0:3], 0' + f]
    if fmt == 'MIMG':
        d = ' dim:SQ_RSRC_IMG_2D'
        return ['v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf' + d, 'v[0:3], v[4:6], s[0:7], s[8:11] dmask:0xf' + d,
                'v[0:3], v[4:7], s[0:7], s[8:11] dmask:0xf' + d, 'v[0:3], v[4:11], s[0:7], s[8:11] dmask:0xf' + d,
                'v[0:3], v[4:5], s[0:7] dmask:0xf' + d, 'v[0:3], v[4:6], s[0:7] dmask:0xf' + d,
                'v[0:3], v4, s[0:7] dmask:0xf' + d, 'v0, v[4:5], s[0:7] dmask:0x1' + d,
                'v[0:1], v[4:5], s[0:7] dmask:0x3' + d, 'v[0:3], v[4:9], s[0:7], s[8:11] dmask:0xf' + d,
                'v[0:3], v[4:10], s[0:7], s[8:11] dmask:0xf' + d, 'v[0:1], v[4:5], s[0:7], s[8:11] dmask:0x3' + d,
                'v0, v[4:5], s[0:7] dmask:0x1 glc' + d, 'v[0:1], v[4:5], s[0:7] dmask:0x3 glc' + d,
                'v[0:3], v[4:12], s[0:7], s[8:11] dmask:0xf' + d,
                'v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1' + d, 'v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1' + d,
                'v[0:3], v[4:7], s[0:7], s[8:11] dmask:0x1' + d, 'v[0:3], v[4:8], s[0:7], s[8:11] dmask:0x1' + d]
    if fmt == 'FLAT':
        return ['v0, v[0:1], off', 'v[0:1], v[0:1], off', 'v[0:2], v[0:1], off', 'v[0:3], v[0:1], off',
                'v[0:1], v2, off', 'v[0:1], v[2:3], off', 'v[0:1], v[2:4], off', 'v[0:1], v[2:5], off',
                'v[0:1], v2, off glc', 'v[0:1], v[2:3], off glc', 'v0, v[0:1]', 'v[0:1], v2']
    raise ValueError(fmt)

def assemble(line):
    p = subprocess.run([LLVM_MC, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64', '-show-encoding'], input=line + '\n',
                       capture_output=True, text=True)
    m = re.search(r'encoding: \[([^\]]*)\]', p.stdout)
    if p.returncode or not m or 'error' in p.stderr:
        return None
    return bytes(int(x, 16) for x in m.group(1).split(','))

def decode(fmt, b):
    w = int.from_bytes(b[0:4], 'little')
    w1 = int.from_bytes(b[4:8], 'little') if len(b) >= 8 else 0
    if fmt == 'SOP2': return (w >> 23) & 0x7f
    if fmt == 'SOPK': return (w >> 23) & 0x1f
    if fmt == 'SOP1': return (w >> 8) & 0xff
    if fmt in ('SOPC', 'SOPP'): return (w >> 16) & 0x7f
    if fmt == 'SMEM': return (w >> 18) & 0xff
    if fmt == 'VOP2': return (w >> 25) & 0x3f
    if fmt == 'VOP1': return (w >> 9) & 0xff
    if fmt == 'VOPC': return (w >> 17) & 0xff
    if fmt == 'VOP3': return (w >> 16) & 0x3ff
    if fmt == 'VOP3P': return (w >> 16) & 0x7f
    if fmt == 'VINTRP': return (w >> 16) & 3
    if fmt == 'DS': return (w >> 18) & 0xff
    if fmt == 'MUBUF': return ((w >> 18) & 0x7f) | (((w >> 25) & 1) << 7)
    if fmt == 'MTBUF': return ((w >> 16) & 7) | (((w1 >> 21) & 1) << 3)
    if fmt == 'MIMG': return ((w >> 18) & 0x7f) | ((w & 1) << 7)
    if fmt == 'FLAT': return (w >> 18) & 0x7f
    raise ValueError(fmt)

def expected_format(b):
    w = int.from_bytes(b[0:4], 'little')
    if (w >> 30) == 2:
        if (w >> 23) == 0x17d: return 'SOP1'
        if (w >> 23) == 0x17e: return 'SOPC'
        if (w >> 23) == 0x17f: return 'SOPP'
        if (w >> 28) == 0xb: return 'SOPK'
        return 'SOP2'
    if (w >> 31) == 0:
        if (w >> 25) == 0x3f: return 'VOP1'
        if (w >> 25) == 0x3e: return 'VOPC'
        return 'VOP2'
    return {0x32: 'VINTRP', 0x33: 'VOP3P', 0x35: 'VOP3', 0x36: 'DS', 0x37: 'FLAT', 0x38: 'MUBUF', 0x3a: 'MTBUF',
            0x3c: 'MIMG', 0x3d: 'SMEM', 0x3e: 'EXP'}.get(w >> 26, '?')

def main():
    groups = [('SOP2', SOP2), ('SOPK', SOPK), ('SOP1', SOP1), ('SOPC', SOPC), ('SOPP', SOPP), ('SMEM', SMEM),
              ('VOP2', VOP2), ('VOP1', VOP1), ('VOPC', VOPC), ('VOP3', VOP3), ('VOP3P', VOP3P), ('VINTRP', VINTRP),
              ('DS', DS), ('MUBUF', MUBUF), ('MTBUF', MTBUF), ('MIMG', MIMG), ('FLAT', FLAT)]
    out = ['// Generated by tools/rdna/gen_ops.py from LLVM\'s gfx1013 assembler; do not edit.',
           '#ifndef KYTY_RDNA_OPS_H', '#define KYTY_RDNA_OPS_H', '', 'namespace KytyWeb::Rdna::Op {', '']
    failed = []
    for fmt, names in groups:
        out.append('enum %s : unsigned {' % fmt)
        seen = {}
        for m in names.split():
            mnem = m
            # VOP3-only encodings of VOP1/VOP2 instructions need the _e64 suffix.
            if fmt == 'VOP3' and m in ('v_cvt_f32_f16',):
                mnem = m + '_e64'
            if fmt == 'VOP3' and m.startswith('v_interp_') and m.endswith('_f32'):
                mnem = m + '_e64'
            enc = None
            for c in candidates(fmt, m):
                b = assemble((mnem + ' ' + c).strip())
                if b and expected_format(b) == fmt:
                    enc = b
                    break
            if enc is None:
                failed.append((fmt, m))
                continue
            op = decode(fmt, enc)
            name = m.upper()
            if fmt == 'VOP3' and m in ('v_cvt_f32_f16', 'v_interp_p1_f32', 'v_interp_p2_f32', 'v_interp_mov_f32'):
                name += '_E64'
            if op in seen:
                print('warning: %s %s and %s share opcode %#x' % (fmt, seen[op], name, op), file=sys.stderr)
            seen[op] = name
            out.append('    %s = 0x%x,' % (name, op))
        out.append('};')
        out.append('')
    out += ['} // namespace KytyWeb::Rdna::Op', '', '#endif', '']
    for f in failed:
        print('could not assemble %s %s' % f, file=sys.stderr)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, 'src/web/include/rdna_ops.h'), 'w') as fh:
        fh.write('\n'.join(out))

main()

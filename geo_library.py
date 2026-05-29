#!/usr/bin/env python3
"""
지질 후보 라이브러리 (Forward-Hypothesis Matching)

11개 지질 패밀리, ~207개 템플릿:
  F1: clean_dip       — 깨끗한 경사 전도층
  F2: covered_dip     — 충적층 피복 경사층
  F3: groundwater     — 경사 지하수/침투 경로
  F4: fault_zone      — 전도성 단층대
  F5: basement        — 경사 기반암 경계
  F6: vertical_block  — 수직 블록 경계
  F7: lens            — 지하수 렌즈
  F8: channel         — 매몰 전도 채널
  F9: composite       — 지하수-단층 복합 구조
 F10: embankment      — 제방/성토체 (침투습윤대·점토코어·수평층)
 F11: basement_fault  — 경사 기반암 경계 + 관통 단층대

각 템플릿: {'family', 'name', 'dip_deg'(None=비단일경사), 'params', 'model_fn'}
"""

import sys
import numpy as np

sys.path.insert(0, '')


# ═══════════════════════════════════════════════════════════
#  모델 빌더
# ═══════════════════════════════════════════════════════════

def _build_clean_dip(mesh, dip_deg, x0, z0, thick, rho_layer, rho_bg):
    """F1: 깨끗한 경사 전도층."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    d = np.radians(dip_deg)
    tan_d = np.tan(d)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = tan_d * (xc - x0) + z0
            if zt <= zc <= zt + thick:
                rho[iz * mesh.ncx + ix] = rho_layer
    return rho


def _build_covered_dip(mesh, dip_deg, x0, z0, thick,
                        cover_thick, rho_layer, rho_cover, rho_bg):
    """F2: 충적층 피복 경사층."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    d = np.radians(dip_deg)
    tan_d = np.tan(d)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            # 피복층 먼저
            if zc <= cover_thick:
                rho[iz * mesh.ncx + ix] = rho_cover
            # 경사 전도층 (피복층보다 우선)
            zt = tan_d * (xc - x0) + z0
            if zt <= zc <= zt + thick:
                rho[iz * mesh.ncx + ix] = rho_layer
    return rho


def _build_groundwater(mesh, dip_deg, x0, z0, width, rho_gw, rho_bg):
    """F3: 경사 지하수/침투 경로 (얇고 매우 전도적)."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    d = np.radians(dip_deg)
    tan_d = np.tan(d)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = tan_d * (xc - x0) + z0
            if zt <= zc <= zt + width:
                rho[iz * mesh.ncx + ix] = rho_gw
    return rho


def _build_fault_zone(mesh, dip_deg, x0, z0, thick, rho_fault, rho_bg):
    """F4: 전도성 단층대 (두꺼운 전도대)."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    d = np.radians(dip_deg)
    tan_d = np.tan(d)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = tan_d * (xc - x0) + z0
            if zt <= zc <= zt + thick:
                rho[iz * mesh.ncx + ix] = rho_fault
    return rho


def _build_basement(mesh, dip_deg, x0, z0, rho_above, rho_below):
    """F5: 경사 기반암 경계 (비저항 경계)."""
    rho = np.full(mesh.n_cells, float(rho_above))
    d = np.radians(dip_deg)
    tan_d = np.tan(d)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = tan_d * (xc - x0) + z0
            if zc > zt:
                rho[iz * mesh.ncx + ix] = rho_below
    return rho


def _build_vertical_block(mesh, x0, width, rho_block, rho_bg):
    """F6: 수직 블록 경계."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]
            if x0 <= xc <= x0 + width:
                rho[iz * mesh.ncx + ix] = rho_block
    return rho


def _build_lens(mesh, xc_l, zc_l, rx, rz, rho_lens, rho_bg):
    """F7: 타원형 지하수 렌즈."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            if ((xc - xc_l) / rx) ** 2 + ((zc - zc_l) / rz) ** 2 <= 1.0:
                rho[iz * mesh.ncx + ix] = rho_lens
    return rho


def _build_channel(mesh, xc_ch, zc_ch, width, height, rho_ch, rho_bg):
    """F8: 매몰 수평 전도 채널."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            if (abs(xc - xc_ch) <= width / 2 and
                    zc_ch <= zc <= zc_ch + height):
                rho[iz * mesh.ncx + ix] = rho_ch
    return rho


def _build_composite(mesh, dip_gw, x0_gw, z0_gw, w_gw,
                      x0_fault, thick_fault,
                      rho_gw, rho_fault, rho_bg):
    """F9: 지하수 경사 경로 + 수직 단층대 복합."""
    rho = np.full(mesh.n_cells, float(rho_bg))
    d_gw = np.radians(dip_gw)
    tan_d = np.tan(d_gw)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt_gw = tan_d * (xc - x0_gw) + z0_gw
            # 지하수 경로
            if zt_gw <= zc <= zt_gw + w_gw:
                rho[iz * mesh.ncx + ix] = rho_gw
            # 단층대 (수직)
            if x0_fault <= xc <= x0_fault + thick_fault:
                # 단층대는 최소 비저항 (지하수와 교차 시 더 낮음)
                cur = rho[iz * mesh.ncx + ix]
                rho[iz * mesh.ncx + ix] = min(cur, rho_fault)
    return rho


def _build_basement_fault(mesh,
                           bm_dip, bm_x0, bm_z0,
                           fault_dip, fault_x0, fault_thick,
                           rho_above, rho_below, rho_fault):
    """F11: 경사 기반암 경계 + 단층대.
    - 배경: 경사진 기반암 경계 (bm_dip, 얕은 풍화대 rho_above / 기반암 rho_below)
    - 단층대: 기반암을 가로지르는 전도성 경사 띠 (fault_dip, fault_thick)
      단층대 비저항은 기반암·풍화대 모두에서 rho_fault로 덮어씀"""
    # 1단계: 경사 기반암 배경
    rho = np.full(mesh.n_cells, float(rho_above))
    bm_tan = np.tan(np.radians(bm_dip))
    ft_tan = np.tan(np.radians(fault_dip))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            z_bm = bm_tan * (xc - bm_x0) + bm_z0
            if zc > z_bm:
                rho[iz * mesh.ncx + ix] = rho_below
    # 2단계: 단층대 덮어쓰기 (기반암 내부까지 관통)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            z_ft = ft_tan * (xc - fault_x0)
            # 단층 중심선에서 수직 거리가 fault_thick/2 이내
            dist = abs(zc - z_ft) / np.sqrt(1 + ft_tan ** 2)
            if dist <= fault_thick / 2.0:
                rho[iz * mesh.ncx + ix] = rho_fault
    return rho


def _build_emb_seepage(mesh, xc, zc, rx, rz, rho_seep, rho_fill):
    """F10-A: 성토체 내부 침투 습윤대 (타원형 저비저항).
    성토체 전체가 rho_fill, 그 안에 타원형 습윤대가 rho_seep."""
    rho = np.full(mesh.n_cells, float(rho_fill))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            x = mesh.x_cc[ix]; z = mesh.z_cc[iz]
            if ((x - xc) / rx) ** 2 + ((z - zc) / rz) ** 2 <= 1.0:
                rho[iz * mesh.ncx + ix] = rho_seep
    return rho


def _build_emb_core(mesh, xc, core_hw, z_core, rho_core, rho_fill):
    """F10-B: 점토 코어 제방 (수직 저비저항 띠).
    양쪽 숄더는 rho_fill, 중앙 코어(폭 2*core_hw)는 rho_core."""
    rho = np.full(mesh.n_cells, float(rho_fill))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            x = mesh.x_cc[ix]; z = mesh.z_cc[iz]
            if abs(x - xc) <= core_hw and z <= z_core:
                rho[iz * mesh.ncx + ix] = rho_core
    return rho


def _build_emb_layers(mesh, z_fill, rho_fill, rho_found):
    """F10-C: 성토층 + 기초지반 수평 2층.
    깊이 z_fill까지 rho_fill, 그 아래 rho_found."""
    rho = np.full(mesh.n_cells, float(rho_fill))
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            z = mesh.z_cc[iz]
            if z > z_fill:
                rho[iz * mesh.ncx + ix] = rho_found
    return rho


# ═══════════════════════════════════════════════════════════
#  템플릿 레지스트리 생성
# ═══════════════════════════════════════════════════════════

def build_template_registry():
    """
    165개 템플릿 리스트 반환.
    각 항목: {family, name, dip_deg, params, builder}
    builder(mesh) → rho_array
    """
    templates = []

    # ── F1: 깨끗한 경사 전도층 (20개) ──────────────────────
    f1_dips  = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    f1_x0s   = [35, 50]
    for dip in f1_dips:
        for x0 in f1_x0s:
            p = dict(dip=dip, x0=x0, z0=1.5, thick=6, rho_layer=20, rho_bg=200)
            templates.append(dict(
                family='clean_dip', name=f'CD_d{dip}_x{x0}',
                dip_deg=float(dip), params=p,
                builder=lambda m, p=p: _build_clean_dip(
                    m, p['dip'], p['x0'], p['z0'],
                    p['thick'], p['rho_layer'], p['rho_bg'])
            ))

    # ── F1 추가: 얕은/깊은 시작 변형 (6개) ─────────────────
    for dip, x0, z0 in [(15, 40, 0.5), (25, 40, 0.5), (35, 40, 0.5),
                         (15, 40, 3.0), (25, 40, 3.0), (35, 40, 3.0)]:
        p = dict(dip=dip, x0=x0, z0=z0, thick=6, rho_layer=20, rho_bg=200)
        templates.append(dict(
            family='clean_dip', name=f'CD_d{dip}_z{z0}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_clean_dip(
                m, p['dip'], p['x0'], p['z0'],
                p['thick'], p['rho_layer'], p['rho_bg'])
        ))

    # ── F2: 피복 경사층 (24개) ──────────────────────────────
    f2_dips    = [10, 15, 20, 25, 30, 35, 40, 45]
    f2_covers  = [2.0, 4.0, 6.0]
    for dip in f2_dips:
        for cov in f2_covers:
            p = dict(dip=dip, x0=40, z0=cov+0.5, thick=6,
                     cover=cov, rho_layer=15, rho_cover=60, rho_bg=200)
            templates.append(dict(
                family='covered_dip', name=f'COV_d{dip}_c{cov}',
                dip_deg=float(dip), params=p,
                builder=lambda m, p=p: _build_covered_dip(
                    m, p['dip'], p['x0'], p['z0'], p['thick'],
                    p['cover'], p['rho_layer'], p['rho_cover'], p['rho_bg'])
            ))

    # ── F3: 경사 지하수 경로 (18개) ────────────────────────
    f3_dips = [5, 10, 15, 20, 25, 30]
    f3_x0s  = [35, 45, 55]
    for dip in f3_dips:
        for x0 in f3_x0s:
            p = dict(dip=dip, x0=x0, z0=1.5, width=3, rho_gw=10, rho_bg=300)
            templates.append(dict(
                family='groundwater', name=f'GW_d{dip}_x{x0}',
                dip_deg=float(dip), params=p,
                builder=lambda m, p=p: _build_groundwater(
                    m, p['dip'], p['x0'], p['z0'],
                    p['width'], p['rho_gw'], p['rho_bg'])
            ))

    # ── F1 추가: 비저항 대비 변형 (6개) ────────────────────
    for dip, rho_l in [(20, 10), (30, 10), (40, 10),
                        (20, 40), (30, 40), (40, 40)]:
        p = dict(dip=dip, x0=40, z0=1.5, thick=6, rho_layer=rho_l, rho_bg=200)
        templates.append(dict(
            family='clean_dip', name=f'CD_d{dip}_rho{rho_l}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_clean_dip(
                m, p['dip'], p['x0'], p['z0'],
                p['thick'], p['rho_layer'], p['rho_bg'])
        ))

    # ── F4: 전도성 단층대 (18개) ────────────────────────────
    f4_dips   = [30, 45, 60, 75]
    f4_x0s    = [30, 40, 55]
    f4_thicks = [8]
    for dip in f4_dips:
        for x0 in f4_x0s:
            p = dict(dip=dip, x0=x0, z0=0.5, thick=8, rho_fault=20, rho_bg=200)
            templates.append(dict(
                family='fault_zone', name=f'FZ_d{dip}_x{x0}',
                dip_deg=float(dip), params=p,
                builder=lambda m, p=p: _build_fault_zone(
                    m, p['dip'], p['x0'], p['z0'],
                    p['thick'], p['rho_fault'], p['rho_bg'])
            ))
    # 얇은 단층대 + 다양한 경사 변형 (7개)
    for dip, x0, thick in [(30, 40, 4), (45, 40, 4), (60, 40, 4),
                             (45, 50, 10), (60, 50, 10),
                             (75, 40, 6), (75, 50, 6)]:
        p = dict(dip=dip, x0=x0, z0=0.5, thick=thick, rho_fault=15, rho_bg=200)
        templates.append(dict(
            family='fault_zone', name=f'FZ_v_d{dip}_x{x0}_t{thick}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_fault_zone(
                m, p['dip'], p['x0'], p['z0'],
                p['thick'], p['rho_fault'], p['rho_bg'])
        ))

    # ── F5: 경사 기반암 경계 (10개) ─────────────────────────
    f5_dips = [5, 10, 15, 20, 25, 30]
    for dip in f5_dips:
        p = dict(dip=dip, x0=40, z0=4.0, rho_above=80, rho_below=800)
        templates.append(dict(
            family='basement', name=f'BM_d{dip}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_basement(
                m, p['dip'], p['x0'], p['z0'],
                p['rho_above'], p['rho_below'])
        ))
    for dip in [12, 22, 35, 45]:
        p = dict(dip=dip, x0=35, z0=3.0, rho_above=100, rho_below=600)
        templates.append(dict(
            family='basement', name=f'BM2_d{dip}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_basement(
                m, p['dip'], p['x0'], p['z0'],
                p['rho_above'], p['rho_below'])
        ))

    # ── F6: 수직 블록 경계 (8개) ────────────────────────────
    for x0 in [25, 35, 45, 55]:
        for width in [5, 10]:
            p = dict(x0=x0, width=width, rho_block=20, rho_bg=200)
            templates.append(dict(
                family='vertical_block', name=f'VB_x{x0}_w{width}',
                dip_deg=None, params=p,
                builder=lambda m, p=p: _build_vertical_block(
                    m, p['x0'], p['width'], p['rho_block'], p['rho_bg'])
            ))

    # ── F7: 지하수 렌즈 (12개) ──────────────────────────────
    for xc in [40, 55, 70]:
        for zc in [3, 5]:
            for rx in [15, 25]:
                p = dict(xc=xc, zc=zc, rx=rx, rz=2.5, rho_lens=15, rho_bg=200)
                templates.append(dict(
                    family='lens', name=f'LN_x{xc}_z{zc}_r{rx}',
                    dip_deg=None, params=p,
                    builder=lambda m, p=p: _build_lens(
                        m, p['xc'], p['zc'], p['rx'], p['rz'],
                        p['rho_lens'], p['rho_bg'])
                ))

    # ── F8: 매몰 채널 (12개) ────────────────────────────────
    for xc in [40, 55, 70]:
        for zc in [2, 4]:
            for w in [10, 20]:
                p = dict(xc=xc, zc=zc, width=w, height=3, rho_ch=15, rho_bg=200)
                templates.append(dict(
                    family='channel', name=f'CH_x{xc}_z{zc}_w{w}',
                    dip_deg=None, params=p,
                    builder=lambda m, p=p: _build_channel(
                        m, p['xc'], p['zc'], p['width'], p['height'],
                        p['rho_ch'], p['rho_bg'])
                ))

    # ── F3 추가: 깊은 지하수 + 피복 변형 (8개) ────────────
    for dip, z0, cover_bg in [(5, 3.0, 400), (10, 3.0, 400),
                               (15, 3.0, 400), (20, 3.0, 400),
                               (5, 5.0, 300), (10, 5.0, 300),
                               (15, 5.0, 300), (20, 5.0, 300)]:
        p = dict(dip=dip, x0=40, z0=z0, width=4, rho_gw=8, rho_bg=cover_bg)
        templates.append(dict(
            family='groundwater', name=f'GW_deep_d{dip}_z{z0}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_groundwater(
                m, p['dip'], p['x0'], p['z0'],
                p['width'], p['rho_gw'], p['rho_bg'])
        ))

    # ── F5 추가: 다양한 기반암 경계 (8개) ──────────────────
    for dip, x0, z0 in [(10, 50, 5.0), (20, 50, 5.0), (30, 50, 5.0),
                         (10, 30, 3.0), (20, 30, 3.0),
                         (5,  40, 6.0), (15, 40, 6.0), (25, 40, 6.0)]:
        p = dict(dip=dip, x0=x0, z0=z0, rho_above=50, rho_below=1000)
        templates.append(dict(
            family='basement', name=f'BM3_d{dip}_x{x0}_z{z0}',
            dip_deg=float(dip), params=p,
            builder=lambda m, p=p: _build_basement(
                m, p['dip'], p['x0'], p['z0'],
                p['rho_above'], p['rho_below'])
        ))

    # ── F6 추가: 고비저항 수직 블록 (4개) ──────────────────
    for x0 in [35, 50]:
        for width in [8, 15]:
            p = dict(x0=x0, width=width, rho_block=800, rho_bg=100)
            templates.append(dict(
                family='vertical_block', name=f'VB_hi_x{x0}_w{width}',
                dip_deg=None, params=p,
                builder=lambda m, p=p: _build_vertical_block(
                    m, p['x0'], p['width'], p['rho_block'], p['rho_bg'])
            ))

    # ── F9: 복합 구조 (10개) ────────────────────────────────
    composites = [
        (10, 30, 1.5, 3, 55, 5),
        (15, 30, 1.5, 3, 55, 5),
        (20, 35, 2.0, 4, 60, 7),
        (25, 35, 2.0, 4, 60, 7),
        (10, 45, 1.5, 3, 40, 5),
        (15, 45, 1.5, 3, 40, 5),
        (20, 50, 2.0, 4, 35, 8),
        (10, 40, 1.0, 3, 65, 5),
        (20, 40, 1.5, 4, 30, 6),
        (30, 35, 2.0, 5, 55, 8),
    ]
    for i, (dip_gw, x0_gw, z0_gw, w_gw, x0_f, t_f) in enumerate(composites):
        p = dict(dip_gw=dip_gw, x0_gw=x0_gw, z0_gw=z0_gw, w_gw=w_gw,
                 x0_fault=x0_f, thick_fault=t_f,
                 rho_gw=10, rho_fault=25, rho_bg=200)
        templates.append(dict(
            family='composite', name=f'COMP_{i+1}',
            dip_deg=float(dip_gw),   # 지하수 경로 기준 경사
            params=p,
            builder=lambda m, p=p: _build_composite(
                m, p['dip_gw'], p['x0_gw'], p['z0_gw'], p['w_gw'],
                p['x0_fault'], p['thick_fault'],
                p['rho_gw'], p['rho_fault'], p['rho_bg'])
        ))

    # ── F10-A: 침투 습윤대 (12개) ───────────────────────────
    # 위치(xc): 측선 중앙(40), 약간 좌(30), 약간 우(55)
    # 깊이(zc): 얕음(3), 깊음(6)
    # 크기(rx,rz): 소(10,3), 대(18,5)
    for xc, zc, rx, rz in [
        (40, 3, 10, 3), (40, 3, 18, 5),
        (40, 6, 10, 3), (40, 6, 18, 5),
        (30, 3, 10, 3), (30, 6, 10, 3),
        (55, 3, 10, 3), (55, 6, 10, 3),
        (40, 4, 14, 4), (40, 8, 14, 4),
        (35, 5, 12, 4), (50, 5, 12, 4),
    ]:
        p = dict(xc=xc, zc=zc, rx=rx, rz=rz, rho_seep=10, rho_fill=80)
        templates.append(dict(
            family='embankment', name=f'EMB_seep_x{xc}_z{zc}_r{rx}',
            dip_deg=None, params=p,
            builder=lambda m, p=p: _build_emb_seepage(
                m, p['xc'], p['zc'], p['rx'], p['rz'],
                p['rho_seep'], p['rho_fill'])
        ))

    # ── F10-B: 점토 코어 (6개) ──────────────────────────────
    for core_hw, z_core in [(5, 15), (8, 15), (12, 15),
                             (5, 10), (8, 10), (12, 10)]:
        p = dict(xc=45, core_hw=core_hw, z_core=z_core,
                 rho_core=8, rho_fill=100)
        templates.append(dict(
            family='embankment', name=f'EMB_core_hw{core_hw}_z{z_core}',
            dip_deg=None, params=p,
            builder=lambda m, p=p: _build_emb_core(
                m, p['xc'], p['core_hw'], p['z_core'],
                p['rho_core'], p['rho_fill'])
        ))

    # ── F10-C: 성토층 + 기초지반 수평층 (4개) ──────────────
    for z_fill, rho_fill, rho_found in [
        (5,  120, 30),   # 고비저항 성토 / 포화 기초
        (8,  120, 30),
        (5,  80,  500),  # 저비저항 성토 / 고비저항 기반암
        (8,  80,  500),
    ]:
        p = dict(z_fill=z_fill, rho_fill=rho_fill, rho_found=rho_found)
        templates.append(dict(
            family='embankment', name=f'EMB_layer_z{z_fill}_rf{rho_fill}',
            dip_deg=None, params=p,
            builder=lambda m, p=p: _build_emb_layers(
                m, p['z_fill'], p['rho_fill'], p['rho_found'])
        ))

    # ── F11: 경사 기반암 + 관통 단층대 (20개) ───────────────
    # 기반암 경사(bm_dip): 5°~25°, 단층 경사(fault_dip): 30°~75°
    # 단층 위치(fault_x0): 측선 내 다양한 위치
    f11_cases = [
        # (bm_dip, bm_x0, bm_z0, fault_dip, fault_x0, fault_thick)
        (5,  40, 4.0, 45, 40, 5),
        (5,  40, 4.0, 60, 40, 5),
        (5,  40, 4.0, 75, 50, 5),
        (10, 40, 4.0, 45, 35, 6),
        (10, 40, 4.0, 60, 45, 6),
        (10, 40, 4.0, 75, 55, 6),
        (15, 40, 3.0, 45, 35, 7),
        (15, 40, 3.0, 60, 45, 7),
        (15, 40, 3.0, 75, 55, 5),
        (20, 40, 3.0, 45, 40, 8),
        (20, 40, 3.0, 60, 50, 6),
        (20, 40, 3.0, 75, 35, 5),
        (25, 40, 3.0, 45, 40, 8),
        (25, 40, 3.0, 60, 50, 6),
        # 얕은 기반암 변형
        (10, 40, 2.5, 60, 40, 5),
        (15, 40, 2.5, 60, 45, 5),
        # 깊은 기반암 변형
        (5,  40, 6.0, 45, 40, 6),
        (10, 40, 6.0, 60, 40, 6),
        # 단층대 얇은 변형
        (10, 40, 4.0, 45, 40, 3),
        (15, 40, 3.0, 60, 45, 3),
    ]
    for (bm_dip, bm_x0, bm_z0, fd, fx0, ft) in f11_cases:
        p = dict(bm_dip=bm_dip, bm_x0=bm_x0, bm_z0=bm_z0,
                 fault_dip=fd, fault_x0=fx0, fault_thick=ft,
                 rho_above=80, rho_below=800, rho_fault=20)
        templates.append(dict(
            family='basement_fault',
            name=f'BMF_bm{bm_dip}_fd{fd}_fx{fx0}',
            dip_deg=float(bm_dip),   # 기반암 경사각 기준
            params=p,
            builder=lambda m, p=p: _build_basement_fault(
                m, p['bm_dip'], p['bm_x0'], p['bm_z0'],
                p['fault_dip'], p['fault_x0'], p['fault_thick'],
                p['rho_above'], p['rho_below'], p['rho_fault'])
        ))

    return templates


# ═══════════════════════════════════════════════════════════
#  빠른 확인
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    templates = build_template_registry()
    families = {}
    for t in templates:
        families.setdefault(t['family'], 0)
        families[t['family']] += 1
    print(f'총 템플릿: {len(templates)}개')
    for fam, cnt in families.items():
        print(f'  {fam:18s}: {cnt}개')

 #!/usr/bin/env python3
"""
RESIS_Pro.py
====================
RESIS Pro - 2D 비저항 탐사 역산 시스템
- APV 파일 (DIPRO 형식) 불러오기
- 전방 모델링 (2.5D 유한차분법)
- 2D 역산 (Gauss-Newton + Tikhonov 정규화)
- GUI 인터페이스 (Tkinter)
"""

import os
import numpy as np
import matplotlib
_RESIS_HEADLESS = os.environ.get('RESIS_HEADLESS', '').lower() in ('1', 'true', 'yes')
matplotlib.use('Agg' if _RESIS_HEADLESS else 'TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
if _RESIS_HEADLESS:
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
else:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
if _RESIS_HEADLESS:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None
else:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(f): return f
        return wrapper


@njit(cache=True)
def _jacobian_core(J, n_data, n_cells, n_ky,
                   c1_idx, c2_idx, p1_idx, p2_idx, K_arr, rho_a,
                   ky_arr, ky_w_arr, sigma,
                   n00, n10, n01, n11, cdx, cdz, carea,
                   phi_all, n_elec_unique, elec_ky_map):
    """자코비안 핵심 루프 (numba JIT 가속)"""
    inv_pi = 1.0 / 3.141592653589793
    for i in range(n_data):
        ic1 = c1_idx[i]; ic2 = c2_idx[i]
        ip1 = p1_idx[i]; ip2 = p2_idx[i]
        Ki = K_arr[i]
        ra_inv = 1.0 / max(abs(rho_a[i]), 1e-10)

        for iky in range(n_ky):
            w = ky_w_arr[iky]
            ky = ky_arr[iky]
            ky2 = ky * ky

            # phi 배열 인덱스: elec_ky_map[elec_idx, iky] → phi_all 오프셋
            off_c1 = elec_ky_map[ic1, iky]
            off_c2 = elec_ky_map[ic2, iky]
            off_p1 = elec_ky_map[ip1, iky]
            off_p2 = elec_ky_map[ip2, iky]

            for j in range(n_cells):
                i00 = n00[j]; i10 = n10[j]; i01 = n01[j]; i11 = n11[j]
                dx = cdx[j]; dz = cdz[j]; area = carea[j]

                # 4개 소스-수신기 조합 (C2P1 - C1P1 - C2P2 + C1P2)
                sk = 0.0
                for combo in range(4):
                    if combo == 0:
                        oi = off_c2; oj = off_p1; sgn = 1.0
                    elif combo == 1:
                        oi = off_c1; oj = off_p1; sgn = -1.0
                    elif combo == 2:
                        oi = off_c2; oj = off_p2; sgn = -1.0
                    else:
                        oi = off_c1; oj = off_p2; sgn = 1.0
                    ps00=phi_all[oi,i00]; ps10=phi_all[oi,i10]
                    ps01=phi_all[oi,i01]; ps11=phi_all[oi,i11]
                    pr00=phi_all[oj,i00]; pr10=phi_all[oj,i10]
                    pr01=phi_all[oj,i01]; pr11=phi_all[oj,i11]
                    gsx = 0.5*((ps10-ps00)+(ps11-ps01))/dx
                    grx = 0.5*((pr10-pr00)+(pr11-pr01))/dx
                    gsz = 0.5*((ps01-ps00)+(ps11-ps10))/dz
                    grz = 0.5*((pr01-pr00)+(pr11-pr10))/dz
                    sa = 0.25*(ps00+ps10+ps01+ps11)
                    ra = 0.25*(pr00+pr10+pr01+pr11)
                    sk += sgn * area * (gsx*grx + gsz*grz + ky2*sa*ra)

                dV_dsig = -(w * inv_pi) * sk
                J[i, j] += Ki * dV_dsig * (-sigma[j]) * ra_inv


# ============================================================
# APV 파일 파서
# ============================================================
def parse_apv(filepath):
    """DIPRO APV 파일을 파싱하여 탐사 정보와 데이터를 반환"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    version = lines[0].strip()
    area = lines[3].strip()
    line_name = lines[4].strip()

    header = lines[5].split()
    n_max = int(header[0])
    n1_count = int(header[1])

    # n=1부터 n=n_max까지의 데이터 개수
    counts = [n1_count - k for k in range(n_max)]
    total_data = sum(counts)

    # 데이터 읽기 (라인 7부터, 0-indexed line 6)
    all_values = []
    idx = 6
    while len(all_values) < total_data and idx < len(lines):
        tokens = lines[idx].split()
        for t in tokens:
            try:
                all_values.append(float(t))
            except ValueError:
                break
        idx += 1
        if len(all_values) >= total_data:
            break

    # 나머지 헤더 파싱 (전극 간격)
    a = None
    for i in range(idx, min(idx + 10, len(lines))):
        line = lines[i].strip()
        try:
            val = float(line)
            if 0.5 < val < 100:  # 합리적인 전극 간격 범위
                a = val
                break
        except ValueError:
            continue

    if a is None:
        a = 5.0  # 기본값

    # n-level별 데이터 분리
    data_by_n = {}
    offset = 0
    for k in range(n_max):
        n = k + 1
        cnt = counts[k]
        data_by_n[n] = np.array(all_values[offset:offset + cnt])
        offset += cnt

    # 전극 수 계산: n=1에서 측정 수 = N_elec - n - 2 + 1 = N_elec - 2
    # n1_count = N_elec - 2 → N_elec = n1_count + 2
    # 실제로는: n=1일때 측정수 = N_dipoles - 1 = (N_elec-1) - 1
    # n1_count = N_elec - 2
    n_electrodes = n1_count + 2 + (n_max - 1)
    # 검증: n=k에서 count = n_electrodes - k - 2
    # n=1: n_electrodes - 3 = n1_count → n_electrodes = n1_count + 3
    # 아니, 다시 계산:
    # 전극 i, i+1 (전류), j=i+n+1, j+1 (전위)
    # j+1 <= N_elec-1 → i+n+2 <= N_elec-1 → i <= N_elec-n-3
    # 측정수 = N_elec-n-3+1 = N_elec-n-2
    # n=1: N_elec - 3 = n1_count → N_elec = n1_count + 3
    n_electrodes = n1_count + 3

    info = {
        'version': version,
        'area': area,
        'line': line_name,
        'a': a,
        'n_max': n_max,
        'n_electrodes': n_electrodes,
        'n1_count': n1_count,
        'data_by_n': data_by_n,
    }
    return info


def parse_res2dinv(filepath):
    """RES2DINV .dat 파일 파싱

    형식:
    Line 1: 측선명
    Line 2: 전극 간격
    Line 3: 배열 유형 (3=dipole-dipole, 1=wenner, 7=wenner-schlumberger)
    Line 4: 총 데이터 수
    Line 5: 위치 유형 (1=첫 전극)
    Line 6: 플래그
    Data: x_c1  a  n  rho_a
    종료: 0 0 0 0
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    line_name = lines[0].strip()
    a = float(lines[1].strip())
    array_code = int(lines[2].strip())
    n_total = int(lines[3].strip())

    # 배열 유형 매핑
    array_map = {1: 'wenner', 3: 'dipole-dipole',
                 7: 'wenner-schlumberger', 6: 'pole-dipole'}
    array_type = array_map.get(array_code, 'dipole-dipole')

    # 데이터 읽기
    data_start = 6  # 보통 7번째 줄부터
    measurements = []
    electrode_positions = set()

    for i in range(data_start, min(data_start + n_total, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4: continue
        try:
            vals = [float(v) for v in parts[:4]]
        except ValueError:
            continue
        if vals[0] == 0 and vals[1] == 0: break

        x_c1 = vals[0]; a_meas = vals[1]; n = int(vals[2]); rho_a = vals[3]

        if array_type == 'dipole-dipole':
            c1 = x_c1; c2 = x_c1 + a_meas
            p1 = c2 + n * a_meas; p2 = p1 + a_meas
        elif array_type == 'wenner':
            c1 = x_c1; p1 = x_c1 + a_meas * n
            p2 = x_c1 + 2 * a_meas * n; c2 = x_c1 + 3 * a_meas * n
        else:
            c1 = x_c1; c2 = x_c1 + a_meas * (2 * n + 1)
            p1 = x_c1 + a_meas * n; p2 = x_c1 + a_meas * (n + 1)

        K = _geometric_factor(c1, c2, p1, p2)
        x_mid = (c1 + c2 + p1 + p2) / 4.0
        z_pseudo = n * a_meas

        measurements.append(dict(
            c1=c1, c2=c2, p1=p1, p2=p2,
            n=n, K=K, x=x_mid, z=z_pseudo, rho_a=rho_a))
        electrode_positions.update([c1, c2, p1, p2])

    electrode_x = np.array(sorted(electrode_positions))
    n_max = max(m['n'] for m in measurements) if measurements else 1
    rho_a_arr = np.array([m['rho_a'] for m in measurements])

    return {
        'line': line_name,
        'area': line_name,
        'a': a,
        'n_max': n_max,
        'n_electrodes': len(electrode_x),
        'array_type': array_type,
        'electrode_x': electrode_x,
        'measurements': measurements,
        'rho_a': rho_a_arr,
    }


def apv_to_survey_data(info, data_type='V_over_I'):
    """APV 데이터를 탐사 배열과 겉보기 비저항으로 변환

    data_type: 'V_over_I' (전위/전류 = 저항) 또는 'rho_a' (겉보기 비저항)
    """
    a = info['a']
    n_max = info['n_max']
    n_elec = info['n_electrodes']
    electrode_x = np.arange(n_elec) * a

    measurements = []
    rho_a_list = []

    for n in range(1, n_max + 1):
        data = info['data_by_n'][n]
        K = np.pi * a * n * (n + 1) * (n + 2)

        for j in range(len(data)):
            c1 = electrode_x[j]
            c2 = electrode_x[j + 1]
            p1 = electrode_x[j + n + 1]
            p2 = electrode_x[j + n + 2]
            x_mid = (c1 + c2 + p1 + p2) / 4.0
            z_pseudo = n * a

            if data_type == 'V_over_I':
                ra = K * data[j]
            else:
                ra = data[j]

            measurements.append(dict(
                c1=c1, c2=c2, p1=p1, p2=p2,
                n=n, K=K, x=x_mid, z=z_pseudo))
            rho_a_list.append(ra)

    return electrode_x, measurements, np.array(rho_a_list)


def filter_bad_data(survey, rho_a, verbose=True):
    """불량 데이터 자동 탐지 및 제거

    1) 음수 또는 0 이하 값 제거
    2) 각 n-level 내에서 중앙값 대비 이상치 제거 (MAD 기반)

    반환: (정제된 survey, 정제된 rho_a, 제거 보고 문자열)
    """
    n_orig = len(rho_a)
    keep = np.ones(n_orig, dtype=bool)
    reasons = []

    # 1) 음수/0 값 제거
    neg_mask = rho_a <= 0
    keep[neg_mask] = False
    if neg_mask.sum() > 0:
        reasons.append(f"  음수/0값: {neg_mask.sum()}개 제거")

    # 2) n-level별 이상치 탐지 (MAD 기반)
    # 주의: 경사 구조 등 이봉분포(bimodal) 데이터에서 과도한 제거 방지
    #   → IQR 대비 MAD가 작으면(이봉분포 징후) 임계값 완화
    n_levels = sorted(set(m['n'] for m in survey.measurements))
    outlier_count = 0
    for n in n_levels:
        indices = [i for i, m in enumerate(survey.measurements) if m['n'] == n and keep[i]]
        if len(indices) < 4:
            continue
        vals = np.log10(rho_a[indices])
        median_v = np.median(vals)
        mad = np.median(np.abs(vals - median_v))
        if mad < 1e-6:
            mad = np.std(vals) * 0.6745  # fallback
        # 이봉분포 감지: IQR/MAD 비율이 크면 분포가 넓다는 의미
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        bimodal_ratio = iqr / max(mad, 1e-6)
        # 일반 정규분포: IQR/MAD ≈ 2.0, 이봉분포: >> 3.0
        if bimodal_ratio > 3.0:
            # 이봉분포: MAD가 극소 → MAD 기반 필터 무의미
            # 대신 IQR 기반 필터 사용 (Tukey's fence: 3×IQR)
            q1, q3 = np.percentile(vals, [25, 75])
            fence = 3.0 * iqr  # 매우 보수적 (표준 Tukey: 1.5 IQR)
            for idx in indices:
                v = np.log10(rho_a[idx])
                if v < q1 - fence or v > q3 + fence:
                    keep[idx] = False
                    outlier_count += 1
        else:
            threshold = 3.5  # 정상 분포: MAD 기반 표준 임계값
            for idx in indices:
                deviation = abs(np.log10(rho_a[idx]) - median_v) / max(mad, 1e-6)
                if deviation > threshold:
                    keep[idx] = False
                    outlier_count += 1
    if outlier_count > 0:
        reasons.append(f"  이상치 (MAD): {outlier_count}개 제거")

    # 정제된 데이터 생성
    new_meas = [m for i, m in enumerate(survey.measurements) if keep[i]]
    new_rho = rho_a[keep]

    new_survey = DipDipSurvey(
        a=survey.a, n_electrodes=survey.n_electrodes,
        n_max=survey.n_max, electrode_x=survey.electrode_x,
        measurements=new_meas)

    n_removed = n_orig - keep.sum()
    report = f"데이터 필터링: {n_orig}개 → {keep.sum()}개 ({n_removed}개 제거)\n"
    report += "\n".join(reasons) if reasons else "  불량 데이터 없음"

    if verbose:
        print(report)

    return new_survey, new_rho, report


# ============================================================
# 탐사 배열 (다중 배열 지원)
# ============================================================

def _geometric_factor(c1, c2, p1, p2):
    """범용 기하학적 인자 K = 2π / G
    G = 1/r(C1,P1) - 1/r(C2,P1) - 1/r(C1,P2) + 1/r(C2,P2)
    compute_data의 dV = phi_c2(P1)-phi_c1(P1)-phi_c2(P2)+phi_c1(P2) 와 부호 호환
    → rho_a = K * dV = 양수 (균질반공간)
    """
    r_c1p1 = abs(c1 - p1); r_c2p1 = abs(c2 - p1)
    r_c1p2 = abs(c1 - p2); r_c2p2 = abs(c2 - p2)
    G = 0.0
    if r_c1p1 > 0: G += 1.0 / r_c1p1
    if r_c2p1 > 0: G -= 1.0 / r_c2p1
    if r_c1p2 > 0: G -= 1.0 / r_c1p2
    if r_c2p2 > 0: G += 1.0 / r_c2p2
    # 음의 부호: compute_data의 dV = phi_c2-phi_c1 관례와 호환
    return -2.0 * np.pi / G if abs(G) > 1e-20 else 1e10


def build_survey_measurements(array_type, electrode_x, a, n_max):
    """배열 유형에 따른 측정 목록 생성

    array_type: 'dipole-dipole', 'wenner', 'wenner-schlumberger', 'pole-dipole'
    """
    e = electrode_x; ne = len(e)
    measurements = []

    if array_type == 'dipole-dipole':
        for i in range(ne - 1):
            for n in range(1, n_max + 1):
                j = i + n + 1
                if j + 1 > ne - 1: break
                c1, c2, p1, p2 = e[i], e[i+1], e[j], e[j+1]
                K = _geometric_factor(c1, c2, p1, p2)
                measurements.append(dict(
                    c1=c1, c2=c2, p1=p1, p2=p2, n=n, K=K,
                    x=(c1+c2+p1+p2)/4.0, z=n*a))

    elif array_type == 'wenner':
        for n in range(1, n_max + 1):
            for i in range(ne - 3*n):
                c1, p1, p2, c2 = e[i], e[i+n], e[i+2*n], e[i+3*n]
                K = _geometric_factor(c1, c2, p1, p2)
                measurements.append(dict(
                    c1=c1, c2=c2, p1=p1, p2=p2, n=n, K=K,
                    x=(c1+c2)/2.0, z=n*a*0.5))

    elif array_type == 'wenner-schlumberger':
        for n in range(1, n_max + 1):
            for i in range(ne - 2*n - 1):
                c1, p1, p2, c2 = e[i], e[i+n], e[i+n+1], e[i+2*n+1]
                K = _geometric_factor(c1, c2, p1, p2)
                measurements.append(dict(
                    c1=c1, c2=c2, p1=p1, p2=p2, n=n, K=K,
                    x=(c1+c2)/2.0, z=n*a*0.7))

    elif array_type == 'pole-dipole':
        for i in range(ne - 1):
            for n in range(1, n_max + 1):
                j = i + n
                if j + 1 > ne - 1: break
                c1, p1, p2 = e[i], e[j], e[j+1]
                c2 = c1 + 1e6  # C2→∞
                K = _geometric_factor(c1, c2, p1, p2)
                measurements.append(dict(
                    c1=c1, c2=c2, p1=p1, p2=p2, n=n, K=K,
                    x=(c1+p1+p2)/3.0, z=n*a*0.5))

    return measurements


class DipDipSurvey:
    """다중 배열 지원 탐사 클래스"""
    def __init__(self, a=10.0, n_electrodes=21, n_max=6,
                 electrode_x=None, measurements=None,
                 array_type='dipole-dipole'):
        self.a = a
        self.n_electrodes = n_electrodes
        self.n_max = n_max
        self.array_type = array_type
        if electrode_x is not None:
            self.electrode_x = electrode_x
        else:
            self.electrode_x = np.arange(n_electrodes) * a
        if measurements is not None:
            self.measurements = measurements
        else:
            self.measurements = build_survey_measurements(
                array_type, self.electrode_x, a, n_max)

    def _build(self):
        self.measurements = build_survey_measurements(
            self.array_type, self.electrode_x, self.a, self.n_max)

    @property
    def n_data(self):
        return len(self.measurements)

    @property
    def unique_electrodes(self):
        pos = set()
        for m in self.measurements:
            pos.update([m['c1'], m['c2'], m['p1'], m['p2']])
        return sorted(pos)


# ============================================================
# 2D 격자
# ============================================================
class Mesh2D:
    def __init__(self, survey, depth_factor=2.5, dx_factor=0.25,
                 n_pad_x=8, n_pad_z=8, pad_factor=1.4, topography=None):
        """topography: 전극 위치에서의 표고 배열 (m), None이면 평탄 지형"""
        a = survey.a
        x0 = survey.electrode_x[0]
        x1 = survey.electrode_x[-1]
        depth = survey.n_max * a * depth_factor
        dx = a * dx_factor

        x_core = np.arange(x0 - 2 * a, x1 + 2 * a + dx * 0.1, dx)
        x_core = np.sort(np.unique(np.concatenate([x_core, survey.electrode_x])))
        z_core = np.arange(0, depth + dx * 0.1, dx)

        xl, xr, zb = [], [], []
        d = dx
        xL, xR, zB = x_core[0], x_core[-1], z_core[-1]
        for _ in range(n_pad_x):
            d *= pad_factor
            xL -= d; xR += d
            xl.insert(0, xL); xr.append(xR)
        d = dx
        for _ in range(n_pad_z):
            d *= pad_factor
            zB += d; zb.append(zB)

        self.x_nodes = np.concatenate([xl, x_core, xr])
        self.z_nodes = np.concatenate([z_core, zb])
        self.nx = len(self.x_nodes)
        self.nz = len(self.z_nodes)
        self.n_nodes = self.nx * self.nz
        self.dx = np.diff(self.x_nodes)
        self.dz = np.diff(self.z_nodes)
        self.ncx = self.nx - 1
        self.ncz = self.nz - 1
        self.n_cells = self.ncx * self.ncz
        self.x_cc = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        self.z_cc = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])
        self.core_x = (x0 - a, x1 + a)
        self.core_z = depth

        # ── 지형 처리 ──
        self.has_topo = topography is not None
        if self.has_topo:
            from scipy.interpolate import interp1d
            topo_f = interp1d(survey.electrode_x, topography,
                              kind='linear', fill_value='extrapolate')
            self.surface_elev = topo_f(self.x_nodes)   # 각 노드 열의 표고
            self.max_elev = self.surface_elev.max()
            # topo_shift: 최고점 대비 각 열이 내려간 깊이
            self.topo_shift = self.max_elev - self.surface_elev
        else:
            self.surface_elev = np.zeros(self.nx)
            self.max_elev = 0.0
            self.topo_shift = np.zeros(self.nx)

        # z_2d[ix, iz]: 절대 z좌표 (최고점 기준, 아래로 양수)
        self.z_2d = self.z_nodes[np.newaxis, :] + self.topo_shift[:, np.newaxis]

    def nidx(self, ix, iz): return iz * self.nx + ix
    def cidx(self, ix, iz): return iz * self.ncx + ix

    def find_node(self, x, z=0.0):
        return np.argmin(np.abs(self.x_nodes - x)), np.argmin(np.abs(self.z_nodes - z))


# ============================================================
# 역산 블록 격자 (전방 격자와 분리)
# ============================================================
class InversionBlocks:
    """깊이 적응형 역산 블록 격자.

    DIPRO/RES2DINV 방식: 깊이가 깊을수록 블록이 커지고,
    데이터 커버리지(사다리꼴) 내부에만 블록을 배치.
    - 1행(천부): 블록 폭 = a, 높이 ≈ a
    - 2행: 블록 폭 = a, 높이 약간 증가
    - k행: 폭 = a × ceil(k/2), 높이 = a × (1 + 0.3*(k-1))
    - 양쪽이 좁아지는 사다리꼴 형태
    """

    def __init__(self, fwd_mesh, survey, depth_increase=0.3):
        a = survey.a
        ex = survey.electrode_x
        self.fwd_mesh = fwd_mesh
        self.a = a

        # 행(깊이) 정의: 각 행의 상단/하단 깊이와 블록 폭
        n_rows = survey.n_max
        z_top = 0.0
        self.rows = []        # list of dict: z_top, z_bot, x_left, x_right, bw, blocks_x
        self.blocks = []      # 전체 블록 리스트: (x_left, z_top, x_right, z_bot, cx, cz)
        block_id = 0

        for row in range(n_rows):
            # 블록 높이: 깊이 증가에 따라 커짐
            bh = a * (1.0 + depth_increase * row)
            z_bot = z_top + bh

            # 블록 폭: 깊이 증가에 따라 커짐 (2행마다 a 추가)
            bw = a * max(1, (row + 2) // 2)

            # x-범위: 사다리꼴 (데이터 커버리지에 맞춤)
            z_mid = 0.5 * (z_top + z_bot)
            x_left = ex[0] + z_mid * 0.5 - a * 0.5
            x_right = ex[-1] - z_mid * 0.5 + a * 0.5

            # 블록 배치
            nx_blocks = max(1, int(np.ceil((x_right - x_left) / bw)))
            actual_bw = (x_right - x_left) / nx_blocks

            row_blocks = []
            for ib in range(nx_blocks):
                bxl = x_left + ib * actual_bw
                bxr = bxl + actual_bw
                bcx = 0.5 * (bxl + bxr)
                bcz = 0.5 * (z_top + z_bot)
                self.blocks.append(dict(
                    xl=bxl, xr=bxr, zt=z_top, zb=z_bot,
                    cx=bcx, cz=bcz, row=row, id=block_id))
                row_blocks.append(block_id)
                block_id += 1

            self.rows.append(dict(
                z_top=z_top, z_bot=z_bot, bw=actual_bw,
                x_left=x_left, x_right=x_right,
                block_ids=row_blocks))
            z_top = z_bot

        self.n_blocks = len(self.blocks)

        # 전방 셀 → 블록 매핑
        fm = fwd_mesh
        self.cell_to_block = np.full(fm.n_cells, -1, dtype=int)
        for ic in range(fm.n_cells):
            ix = ic % fm.ncx; iz = ic // fm.ncx
            xc, zc = fm.x_cc[ix], fm.z_cc[iz]
            for bi, blk in enumerate(self.blocks):
                if blk['xl'] <= xc < blk['xr'] and blk['zt'] <= zc < blk['zb']:
                    self.cell_to_block[ic] = bi
                    break

    def blocks_to_cells(self, block_rho):
        """블록 비저항 → 전방 셀 비저항 매핑"""
        default = np.median(block_rho) if len(block_rho) > 0 else 100.0
        cell_rho = np.full(self.fwd_mesh.n_cells, default)
        for ic in range(self.fwd_mesh.n_cells):
            bi = self.cell_to_block[ic]
            if 0 <= bi < len(block_rho):
                cell_rho[ic] = block_rho[bi]
        return cell_rho

    def cells_to_blocks_jacobian(self, J_cells):
        """셀 자코비안 → 블록 자코비안 (합산)"""
        nd = J_cells.shape[0]
        J_b = np.zeros((nd, self.n_blocks))
        for ic in range(self.fwd_mesh.n_cells):
            bi = self.cell_to_block[ic]
            if 0 <= bi < self.n_blocks:
                J_b[:, bi] += J_cells[:, ic]
        return J_b

    def build_reg_matrices(self, alpha_s=0.01, alpha_x=1.0, alpha_z=1.0):
        """블록 기반 정규화 행렬 생성"""
        nb = self.n_blocks
        Ws = sparse.eye(nb) * alpha_s

        # x-평활: 같은 행 내 인접 블록
        rx, cx, vx = [], [], []
        idx = 0
        for row_info in self.rows:
            bids = row_info['block_ids']
            for k in range(len(bids) - 1):
                b1, b2 = bids[k], bids[k + 1]
                dx = self.blocks[b2]['cx'] - self.blocks[b1]['cx']
                if dx > 0:
                    rx.extend([idx, idx]); cx.extend([b1, b2])
                    vx.extend([-1.0 / dx, 1.0 / dx]); idx += 1
        Wx = sparse.csr_matrix((vx, (rx, cx)), shape=(idx, nb)) * alpha_x if idx > 0 else sparse.csr_matrix((0, nb))

        # z-평활: 인접 행 간 수직으로 가까운 블록
        rz, cz, vz = [], [], []
        idx = 0
        for r in range(len(self.rows) - 1):
            for bi in self.rows[r]['block_ids']:
                bcx = self.blocks[bi]['cx']
                # 아래 행에서 가장 가까운 블록 찾기
                best_bj, best_dist = -1, 1e10
                for bj in self.rows[r + 1]['block_ids']:
                    dist = abs(self.blocks[bj]['cx'] - bcx)
                    if dist < best_dist:
                        best_dist = dist; best_bj = bj
                if best_bj >= 0:
                    dz = self.blocks[best_bj]['cz'] - self.blocks[bi]['cz']
                    if dz > 0:
                        rz.extend([idx, idx]); cz.extend([bi, best_bj])
                        vz.extend([-1.0 / dz, 1.0 / dz]); idx += 1
        Wz = sparse.csr_matrix((vz, (rz, cz)), shape=(idx, nb)) * alpha_z if idx > 0 else sparse.csr_matrix((0, nb))

        return Ws, Wx, Wz

    def plot_blocks(self, ax, block_rho=None, cmap=None, norm=None,
                    show_values=False, show_grid_only=False):
        """블록 격자를 ax에 그리기
        show_grid_only=True: 보간 위에 블록 경계선만 얇게 오버레이
        """
        if cmap is None:
            cmap = RHO_CMAP
        for bi, blk in enumerate(self.blocks):
            xl, xr = blk['xl'], blk['xr']
            zt, zb = blk['zt'], blk['zb']
            if show_grid_only:
                rect = Rectangle((xl, zt), xr - xl, zb - zt,
                                 facecolor='none', edgecolor='white',
                                 linewidth=0.5, alpha=0.6, zorder=9)
                ax.add_patch(rect)
            else:
                rv = block_rho[bi] if block_rho is not None else 100.0
                fc = cmap(norm(rv)) if norm else 'lightgray'
                rect = Rectangle((xl, zt), xr - xl, zb - zt,
                                 facecolor=fc, edgecolor='gray',
                                 linewidth=0.3, zorder=3)
                ax.add_patch(rect)
                if show_values and (xr - xl) > self.a * 0.8:
                    ax.text(blk['cx'], blk['cz'], f'{rv:.0f}',
                            fontsize=4, ha='center', va='center', zorder=6)


# ============================================================
# 2.5D 전방 모델링
# ============================================================
class ForwardSolver:
    def __init__(self, mesh, rho):
        self.mesh = mesh
        self.rho = np.asarray(rho, dtype=float)
        self.sigma = 1.0 / self.rho
        self._sigma_node = self._avg_to_nodes()
        self._setup_ky()
        self.phi_ky_cache = {}

    def _avg_to_nodes(self):
        """셀 전도도 → 노드 전도도 (면적 가중 조화평균, 벡터화)

        조화평균은 전류 흐름에 대한 직렬 저항 모델로,
        전도도 경계(σ₁≠σ₂)에서 산술평균보다 물리적으로 정확하다.
        산술: σ_node = Σ(σᵢAᵢ)/Σ(Aᵢ)         → 경계 번짐
        조화: σ_node = Σ(Aᵢ) / Σ(Aᵢ/σᵢ)       → 경계 보존

        참고: Dey & Morrison (1979), 셀 간 전도도 보간
        """
        m = self.mesh
        ix_arr, iz_arr = np.meshgrid(np.arange(m.ncx), np.arange(m.ncz))
        ix_f = ix_arr.ravel(); iz_f = iz_arr.ravel()
        sigma_f = self.sigma[iz_f * m.ncx + ix_f]
        area_f = m.dx[ix_f] * m.dz[iz_f]

        # 조화평균: Σ(Aᵢ) / Σ(Aᵢ/σᵢ)
        inv_sigma_a = area_f / np.maximum(sigma_f, 1e-30)

        sum_area = np.zeros(m.n_nodes)
        sum_inv = np.zeros(m.n_nodes)
        for diz in (0, 1):
            for dix in (0, 1):
                ni = (iz_f + diz) * m.nx + (ix_f + dix)
                np.add.at(sum_area, ni, area_f)
                np.add.at(sum_inv, ni, inv_sigma_a)
        return sum_area / np.maximum(sum_inv, 1e-30)

    def _setup_ky(self):
        m = self.mesh
        L = m.x_nodes[-1] - m.x_nodes[0]
        pts, wts = np.polynomial.legendre.leggauss(9)
        lmin, lmax = np.log(0.3 / L), np.log(3.0 / m.dx.min())
        lk = 0.5 * (lmax - lmin) * pts + 0.5 * (lmax + lmin)
        self.ky = np.exp(lk); self.ky_w = wts * 0.5 * (lmax - lmin) * self.ky

    def _build_A(self, ky):
        """FD 시스템 행렬 조립 (벡터화)"""
        m = self.mesh; nx, nz = m.nx, m.nz; sn = self._sigma_node
        all_r, all_c, all_v = [], [], []

        # ── 경계 노드: Dirichlet φ=0 ──
        # 좌/우 경계
        for bx in [0, nx - 1]:
            ni_bc = np.arange(nz) * nx + bx
            all_r.append(ni_bc); all_c.append(ni_bc)
            all_v.append(np.ones(nz))
        # 하단 경계
        ni_bot = (nz - 1) * nx + np.arange(1, nx - 1)
        all_r.append(ni_bot); all_c.append(ni_bot)
        all_v.append(np.ones(len(ni_bot)))

        # ── 내부 노드 (iz=1..nz-2) ──
        ix_int = np.arange(1, nx - 1)
        for iz in range(1, nz - 1):
            ni = iz * nx + ix_int
            hxw = m.x_nodes[ix_int] - m.x_nodes[ix_int - 1]
            hxe = m.x_nodes[ix_int + 1] - m.x_nodes[ix_int]
            hxa = 0.5 * (hxw + hxe)
            hzu = m.z_nodes[iz] - m.z_nodes[iz - 1]
            hzd = m.z_nodes[iz + 1] - m.z_nodes[iz]
            hza = 0.5 * (hzu + hzd)

            sw = 0.5 * (sn[ni] + sn[ni - 1])
            se = 0.5 * (sn[ni] + sn[ni + 1])
            su = 0.5 * (sn[ni] + sn[ni - nx])
            sd = 0.5 * (sn[ni] + sn[ni + nx])
            cw = sw / (hxw * hxa)
            ce = se / (hxe * hxa)
            cu = su / (hzu * hza)
            cd = sd / (hzd * hza)
            diag = -(cw + ce + cu + cd + ky ** 2 * sn[ni])

            all_r.extend([ni, ni, ni, ni, ni])
            all_c.extend([ni - 1, ni + 1, ni - nx, ni + nx, ni])
            all_v.extend([cw, ce, cu, cd, diag])

        # ── 표면 노드 (iz=0): Neumann BC ──
        iz = 0
        ni = ix_int.copy()  # iz=0이므로 ni = ix
        hxw = m.x_nodes[ix_int] - m.x_nodes[ix_int - 1]
        hxe = m.x_nodes[ix_int + 1] - m.x_nodes[ix_int]
        hxa = 0.5 * (hxw + hxe)
        hzd = m.z_nodes[1] - m.z_nodes[0]
        hza = hzd * 0.5  # 반셀

        sw = 0.5 * (sn[ni] + sn[ni - 1])
        se = 0.5 * (sn[ni] + sn[ni + 1])
        sd = 0.5 * (sn[ni] + sn[ni + nx])
        cw = sw / (hxw * hxa)
        ce = se / (hxe * hxa)
        cd = sd / (hzd * hza)
        diag = -(cw + ce + cd + ky ** 2 * sn[ni])

        all_r.extend([ni, ni, ni, ni])
        all_c.extend([ni - 1, ni + 1, ni + nx, ni])
        all_v.extend([cw, ce, cd, diag])

        rows = np.concatenate(all_r)
        cols = np.concatenate(all_c)
        vals = np.concatenate(all_v)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(m.n_nodes, m.n_nodes))

    def compute_data(self, survey, callback=None):
        """LU 분해 재사용: 각 ky마다 1회 분해 → 모든 소스 back-solve"""
        from scipy.sparse.linalg import splu
        self.phi_ky_cache = {}
        m = self.mesh
        elecs = survey.unique_electrodes

        # 소스 노드 + 제어체적 사전 계산
        src_info = {}
        for sx in elecs:
            six, siz = m.find_node(sx, 0.0)
            sni = m.nidx(six, siz)
            if 0 < six < m.nx - 1:
                dx_cv = 0.5 * (m.dx[six - 1] + m.dx[six])
            else:
                dx_cv = m.dx[min(six, m.ncx - 1)]
            dz_cv = m.dz[0] * 0.5 if siz == 0 else (
                0.5 * (m.dz[siz - 1] + m.dz[siz]) if siz < m.nz - 1 else m.dz[-1])
            src_info[sx] = (sni, six, siz, dx_cv * dz_cv)

        phi3d = {sx: np.zeros(m.n_nodes) for sx in elecs}

        for iky, (ky, w) in enumerate(zip(self.ky, self.ky_w)):
            A = self._build_A(ky)
            # LU 분해 1회
            lu = splu(A.tocsc())

            # 모든 소스에 대해 back-solve (빠름)
            for sx in elecs:
                sni, six, siz, V_cv = src_info[sx]
                rhs = np.zeros(m.n_nodes)
                if 0 < six < m.nx - 1 and siz < m.nz - 1:
                    rhs[sni] = -1.0 / V_cv
                phi_ky = lu.solve(rhs)
                self.phi_ky_cache[(sx, iky)] = phi_ky.copy()
                phi3d[sx] += w * phi_ky / np.pi

            if callback:
                callback(iky + 1, len(self.ky))
        m = self.mesh; rho_a = np.zeros(survey.n_data)
        for i, ms in enumerate(survey.measurements):
            p1i = m.nidx(*m.find_node(ms['p1']))
            p2i = m.nidx(*m.find_node(ms['p2']))
            dV = (phi3d[ms['c2']][p1i] - phi3d[ms['c1']][p1i]
                  - phi3d[ms['c2']][p2i] + phi3d[ms['c1']][p2i])
            rho_a[i] = ms['K'] * dV
        return rho_a


# ============================================================
# 2.5D FEM 전방 모델링 (삼각형 비정형 격자)
# ============================================================
class TriMesh:
    """Delaunay 삼각형 격자 (지형 대응)"""

    def __init__(self, survey, depth_factor=2.5, dx_factor=0.5,
                 n_pad=6, pad_factor=1.5, topography=None):
        from scipy.spatial import Delaunay
        a = survey.a
        ex = survey.electrode_x
        x0, x1 = ex[0], ex[-1]
        depth = survey.n_max * a * depth_factor
        dx = a * dx_factor

        # 지형
        self.has_topo = topography is not None
        if self.has_topo:
            from scipy.interpolate import interp1d
            topo_f = interp1d(ex, topography, kind='linear', fill_value='extrapolate')
        else:
            topo_f = lambda x: np.zeros_like(np.atleast_1d(x))

        # 내부 노드 생성 (지형 따라 배치)
        x_core = np.arange(x0 - 2 * a, x1 + 2 * a + dx * 0.1, dx)
        x_core = np.sort(np.unique(np.concatenate([x_core, ex])))
        # 패딩
        x_pad_l, x_pad_r = [], []
        d = dx; xL, xR = x_core[0], x_core[-1]
        for _ in range(n_pad):
            d *= pad_factor; xL -= d; xR += d
            x_pad_l.insert(0, xL); x_pad_r.append(xR)
        x_all = np.concatenate([x_pad_l, x_core, x_pad_r])

        z_layers = [0]
        d = dx
        while z_layers[-1] < depth:
            z_layers.append(z_layers[-1] + d)
            d *= 1.1
        # 패딩 아래
        for _ in range(n_pad):
            d *= pad_factor; z_layers.append(z_layers[-1] + d)
        z_layers = np.array(z_layers)

        # 2D 노드 좌표 생성 (x, z_absolute)
        pts = []
        for xi in x_all:
            surf = float(topo_f(xi))
            for zj in z_layers:
                pts.append([xi, surf - zj])  # z = elevation - depth
        pts = np.array(pts)
        self.max_elev = float(topo_f(x_all).max()) if self.has_topo else 0.0

        # Delaunay 삼각분할
        tri = Delaunay(pts)
        self.nodes = pts                     # (n_nodes, 2): [x, elevation]
        self.elements = tri.simplices        # (n_elem, 3): 노드 인덱스
        self.n_nodes = len(pts)
        self.n_elements = len(self.elements)
        self.x_all = x_all
        self.z_layers = z_layers
        self.n_z = len(z_layers)

        # 전극 → 가장 가까운 노드 매핑
        self.elec_node = {}
        for xe in ex:
            surf = float(topo_f(xe))
            dists = np.sqrt((pts[:, 0] - xe) ** 2 + (pts[:, 1] - surf) ** 2)
            self.elec_node[xe] = int(np.argmin(dists))

        # 요소 중심좌표 (비저항 할당용)
        self.elem_centers = pts[self.elements].mean(axis=1)
        # 요소 면적
        self._compute_areas()

        # 경계 노드: 좌/우/하단만 (표면은 Neumann → 제외)
        hull = set()
        for s in tri.convex_hull:
            hull.update(s)
        # 표면 노드(최상위 z) 제외: 지표면 근처 노드는 경계에서 빼줌
        surface_tol = max(z_layers[1] * 0.6, a * 0.3)
        surface_elevs = np.array([float(topo_f(p[0])) for p in pts])
        boundary = []
        for ni in sorted(hull):
            if pts[ni, 1] < surface_elevs[ni] - surface_tol:
                boundary.append(ni)  # 표면 아래: 좌/우/하단 경계
        # 양 끝 패딩의 표면 노드도 경계에 포함 (먼 거리)
        x_range = x_all[-1] - x_all[0]
        for ni in sorted(hull):
            if pts[ni, 0] < x_all[0] + a or pts[ni, 0] > x_all[-1] - a:
                if ni not in boundary:
                    boundary.append(ni)
        self.boundary_nodes = np.array(sorted(set(boundary)))

    def _compute_areas(self):
        p = self.nodes[self.elements]
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        self.areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


class ForwardFEM:
    """2.5D FEM 전방 모델링 (삼각형 격자)"""

    def __init__(self, tri_mesh, rho):
        self.mesh = tri_mesh
        self.rho = np.asarray(rho, dtype=float)
        self.sigma = 1.0 / self.rho
        self._setup_ky()
        self.phi_ky_cache = {}

    def _setup_ky(self):
        m = self.mesh
        L = np.ptp(m.nodes[:, 0])
        dx_min = np.min(np.sqrt(m.areas)) * 0.5
        pts, wts = np.polynomial.legendre.leggauss(9)
        lmin, lmax = np.log(0.3 / L), np.log(3.0 / max(dx_min, 0.1))
        lk = 0.5 * (lmax - lmin) * pts + 0.5 * (lmax + lmin)
        self.ky = np.exp(lk)
        self.ky_w = wts * 0.5 * (lmax - lmin) * self.ky

    def _build_stiffness(self, ky):
        """글로벌 강성행렬 조립 (완전 벡터화)"""
        m = self.mesh
        n = m.n_nodes; ne = m.n_elements
        el = m.elements                          # (ne, 3)
        coords = m.nodes[el]                     # (ne, 3, 2)
        sigma = self.sigma                       # (ne,)
        area = m.areas                           # (ne,)

        # 유효 요소만 (면적 > 0)
        valid = area > 1e-20
        el_v = el[valid]; coords_v = coords[valid]
        sigma_v = sigma[valid]; area_v = area[valid]

        x = coords_v[:, :, 0]; y = coords_v[:, :, 1]  # (nv, 3)
        # 형상함수 기울기 b, c: (nv, 3)
        b = np.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]])
        c = np.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]])

        # 9개 (i,j) 조합의 행/열/값 한꺼번에 계산
        rows_all, cols_all, vals_all = [], [], []
        for i in range(3):
            for j in range(3):
                k_diff = sigma_v / (4.0 * area_v) * (b[:, i] * b[:, j] + c[:, i] * c[:, j])
                k_mass = ky ** 2 * sigma_v * area_v * (1.0 / 6.0 if i == j else 1.0 / 12.0)
                rows_all.append(el_v[:, i])
                cols_all.append(el_v[:, j])
                vals_all.append(k_diff + k_mass)

        rows_arr = np.concatenate(rows_all)
        cols_arr = np.concatenate(cols_all)
        vals_arr = np.concatenate(vals_all)
        K = sparse.csr_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n, n))

        # 경계조건 (Dirichlet φ=0) — 대각 penalization (빠름)
        bn = m.boundary_nodes
        big = 1e20
        K = K.tolil()
        for bi in bn:
            K[bi, bi] = big
        return K.tocsr()

    def solve_source(self, sx):
        m = self.mesh
        src_ni = m.elec_node[sx]
        phi = np.zeros(m.n_nodes)

        for iky, (ky, w) in enumerate(zip(self.ky, self.ky_w)):
            K = self._build_stiffness(ky)
            rhs = np.zeros(m.n_nodes)
            if src_ni not in m.boundary_nodes:
                rhs[src_ni] = 1.0  # FEM 약형식: 소스항 양수
            phi_ky = spsolve(K, rhs)
            self.phi_ky_cache[(sx, iky)] = phi_ky.copy()
            phi += w * phi_ky / np.pi
        return phi

    def compute_data(self, survey, callback=None):
        self.phi_ky_cache = {}
        m = self.mesh
        elecs = survey.unique_electrodes
        phi3d = {}
        for ie, ex in enumerate(elecs):
            phi3d[ex] = self.solve_source(ex)
            if callback: callback(ie + 1, len(elecs))

        rho_a = np.zeros(survey.n_data)
        for i, ms in enumerate(survey.measurements):
            p1i = m.elec_node[ms['p1']]
            p2i = m.elec_node[ms['p2']]
            dV = (phi3d[ms['c2']][p1i] - phi3d[ms['c1']][p1i]
                  - phi3d[ms['c2']][p2i] + phi3d[ms['c1']][p2i])
            rho_a[i] = ms['K'] * dV
        return rho_a


class _FEMtoFDMWrapper:
    """FEM solver를 FDM 자코비안과 호환되도록 래핑.
    FEM 전위를 FDM 노드 격자로 보간하여 phi_ky_cache 제공."""

    def __init__(self, fem_solver, fdm_mesh, tri_mesh):
        self.fem = fem_solver
        self.fdm = fdm_mesh
        self.tri = tri_mesh
        self.sigma = np.ones(fdm_mesh.n_cells)
        self.ky = fem_solver.ky
        self.ky_w = fem_solver.ky_w
        self.phi_ky_cache = {}

        # 보간 매핑 사전 계산 (FEM→FDM 최근접 노드)
        fem_pts = tri_mesh.nodes.copy()
        fem_pts[:, 1] = tri_mesh.max_elev - fem_pts[:, 1]  # 표고→깊이
        fdm_x = np.repeat(fdm_mesh.x_nodes, fdm_mesh.nz)
        fdm_z = fdm_mesh.z_2d.ravel()
        # 각 FDM 노드에 대해 가장 가까운 FEM 노드 3개의 가중 평균
        from scipy.spatial import cKDTree
        tree = cKDTree(fem_pts)
        dists, idxs = tree.query(np.column_stack([fdm_x, fdm_z]), k=3)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        weights /= weights.sum(axis=1, keepdims=True)
        self._interp_idx = idxs    # (n_fdm_nodes, 3)
        self._interp_wt = weights   # (n_fdm_nodes, 3)

    def _interp_to_fdm(self, phi_fem):
        """FEM 전위 → FDM 전위 (사전 계산된 IDW 보간)"""
        phi_neighbors = phi_fem[self._interp_idx]   # (n_fdm, 3)
        return np.sum(phi_neighbors * self._interp_wt, axis=1)

    def compute_data(self, survey, callback=None):
        rho_a = self.fem.compute_data(survey, callback=callback)
        self.phi_ky_cache = {}
        for (sx, iky), phi_fem in self.fem.phi_ky_cache.items():
            self.phi_ky_cache[(sx, iky)] = self._interp_to_fdm(phi_fem)
        return rho_a


# ============================================================
# DOI (Depth of Investigation) 지수
# ============================================================
def compute_doi(survey, mesh, d_obs, rho_ref, callback=None,
                auto_alpha=True, robust=True, max_iter=5,
                solver_type='FDM', tri_mesh=None):
    """Oldenburg & Li (1999) DOI 지수 계산

    두 가지 다른 기준 모델로 역산하여 결과 차이로 신뢰도 평가.
    DOI = 0 → 완전 신뢰, DOI = 1 → 신뢰 불가
    """
    rho1 = rho_ref * 0.1   # 기준 모델 1: 낮은 비저항
    rho2 = rho_ref * 10.0  # 기준 모델 2: 높은 비저항

    if callback: callback("DOI: 기준 모델 1 (저비저항) 역산...")
    inv1 = Inversion2D(survey, mesh, rho_ref=rho1,
                        alpha=1.0, max_iter=max_iter, tol=0.05,
                        solver_type=solver_type, tri_mesh=tri_mesh)
    m1, _, _ = inv1.run(d_obs, auto_alpha=auto_alpha, robust=robust,
                         callback=callback)

    if callback: callback("DOI: 기준 모델 2 (고비저항) 역산...")
    inv2 = Inversion2D(survey, mesh, rho_ref=rho2,
                        alpha=1.0, max_iter=max_iter, tol=0.05,
                        solver_type=solver_type, tri_mesh=tri_mesh)
    m2, _, _ = inv2.run(d_obs, auto_alpha=auto_alpha, robust=robust,
                         callback=callback)

    # DOI 계산
    log_m1 = np.log10(m1)
    log_m2 = np.log10(m2)
    log_ref1 = np.log10(rho1)
    log_ref2 = np.log10(rho2)

    denom = log_ref1 - log_ref2
    doi = np.abs(log_m1 - log_m2) / max(abs(denom), 1e-10)
    doi = np.clip(doi, 0, 1)

    if callback: callback(f"DOI 완료: 신뢰 영역(DOI<0.3) = {(doi<0.3).sum()}/{len(doi)} 셀")
    return doi


# ============================================================
# 2D 역산
# ============================================================
class Inversion2D:
    def __init__(self, survey, mesh, rho_ref=100.0,
                 alpha=1.0, alpha_s=0.01, alpha_x=1.0, alpha_z=1.0,
                 max_iter=8, tol=0.02, solver_type='FDM', tri_mesh=None,
                 use_blocks=True, reg_type='L2', mgs_beta=0.5,
                 dip_angle=0.0, dip_weight=1.0,
                 noise_floor=0.001, pct_error=0.05, target_chi2=1.0,
                 cooling_factor=0.7, min_alpha=1e-4,
                 ridge_ratio=1e-3, record_condition=False,
                 mgs_alpha_factor_cell=0.3, mgs_alpha_factor_block=0.1,
                 mgs_blending=True, mgs_weight_clip=0.01,
                 mgs_divergence_threshold=1.3,
                 aniso_proper=True,
                 use_structure_tensor=False,
                 st_dip_weight=3.0,
                 st_update_interval=2,
                 st_smooth_sigma=2.0,
                 initial_model=None,
                 reference_model=None,
                 skip_phase1_if_initial=True):
        """
        use_blocks: True=깊이적응형 블록, False=전방격자 셀 직접
        reg_type: 'L2' (Tikhonov), 'MGS' (Minimum Gradient Support)
        mgs_beta: MGS 임계값 (작을수록 날카로운 경계 허용)
        dip_angle: 이방성 정규화 경사각 (도, 0=수평, 양수=오른쪽 하향)
        dip_weight: 경사 방향 평활화 강도 (>1: 경사 따라 더 평활, 1=등방)

        --- 학술적 통계 프레임워크 (Occam's inversion) ---
        noise_floor: 절대 노이즈 하한 (log 도메인, Kemna 2000)
        pct_error: 상대 오차 비율 (5% = 0.05, LaBrecque et al. 1996)
        target_chi2: 목표 chi-squared/N (Constable et al. 1987, 이상적 = 1.0)
        cooling_factor: α 냉각 비율 (0<f<1, Occam 전략)
        min_alpha: α 하한 (과적합 방지)
        """
        self.survey = survey; self.mesh = mesh
        self.rho_ref = rho_ref; self.alpha = alpha
        self.alpha_s = alpha_s; self.alpha_x = alpha_x
        self.alpha_z = alpha_z; self.max_iter = max_iter; self.tol = tol
        self.solver_type = solver_type
        self.tri_mesh = tri_mesh
        self.use_blocks = use_blocks
        self.reg_type = reg_type
        self.mgs_beta = mgs_beta
        self.dip_angle = dip_angle
        self.dip_weight = dip_weight
        # 통계적 데이터 오차 모델
        self.noise_floor = noise_floor
        self.pct_error = pct_error
        self.target_chi2 = target_chi2
        self.cooling_factor = cooling_factor
        self.min_alpha = min_alpha
        # 실험용 계측 파라미터
        self.ridge_ratio = ridge_ratio
        self.record_condition = record_condition
        self.cond_history = []
        # ablation knobs (모두 default = 현재 동작 유지)
        self.mgs_alpha_factor_cell = mgs_alpha_factor_cell
        self.mgs_alpha_factor_block = mgs_alpha_factor_block
        self.mgs_blending = mgs_blending
        self.mgs_weight_clip = mgs_weight_clip
        self.mgs_divergence_threshold = mgs_divergence_threshold
        self.aniso_proper = aniso_proper
        # 구조 텐서 적응형 정규화 (Structure Tensor Adaptive Regularization)
        self.use_structure_tensor = use_structure_tensor
        self.st_dip_weight = st_dip_weight
        self.st_update_interval = st_update_interval
        self.st_smooth_sigma = st_smooth_sigma
        self._st_dip_field = None   # 마지막으로 추정된 국소 경사각 필드 저장
        self._st_confidence_field = None   # 구조 텐서 coherence 기반 국소 신뢰도(0~1)
        self.initial_model = initial_model
        self.reference_model = reference_model
        self.skip_phase1_if_initial = skip_phase1_if_initial
        # 확장 수렴 이력 (학술 분석용)
        self.convergence = {
            'rms': [], 'chi2': [], 'roughness': [],
            'alpha': [], 'phi_d': [], 'phi_m': []
        }
        self._last_J = None  # 감도/분해능 분석용 자코비안 저장

        # 블록 격자 생성
        if use_blocks:
            self.inv_blocks = InversionBlocks(mesh, survey)
            self.n_params = self.inv_blocks.n_blocks
        else:
            self.inv_blocks = None
            self.n_params = mesh.n_cells
        self._build_reg()

    def _build_reg(self):
        if self.use_blocks and self.inv_blocks is not None:
            self.Ws, self.Wx, self.Wz = self.inv_blocks.build_reg_matrices(
                self.alpha_s, self.alpha_x, self.alpha_z)
            # 블록 모드에서도 이방성 적용
            if abs(self.dip_angle) > 0.1 and self.dip_weight > 1.0:
                self._apply_anisotropic_rotation()
            return

        m = self.mesh; nc = m.n_cells
        self.Ws = sparse.eye(nc) * self.alpha_s
        r, c, v = [], [], []; idx = 0
        for iz in range(m.ncz):
            for ix in range(m.ncx - 1):
                dx = m.x_cc[ix + 1] - m.x_cc[ix]
                r.append(idx); c.append(m.cidx(ix, iz)); v.append(-1.0 / dx)
                r.append(idx); c.append(m.cidx(ix + 1, iz)); v.append(1.0 / dx); idx += 1
        self.Wx = sparse.csr_matrix((v, (r, c)), shape=(idx, nc)) * self.alpha_x
        r, c, v = [], [], []; idx = 0
        for iz in range(m.ncz - 1):
            for ix in range(m.ncx):
                dz = m.z_cc[iz + 1] - m.z_cc[iz]
                r.append(idx); c.append(m.cidx(ix, iz)); v.append(-1.0 / dz)
                r.append(idx); c.append(m.cidx(ix, iz + 1)); v.append(1.0 / dz); idx += 1
        self.Wz = sparse.csr_matrix((v, (r, c)), shape=(idx, nc)) * self.alpha_z

        # 이방성 정규화: 경사 방향 회전 적용
        if abs(self.dip_angle) > 0.1 and self.dip_weight > 1.0:
            self._apply_anisotropic_rotation()

    def _build_cell_center_gradients(self):
        """셀 중심에서 정의된 ∂/∂x, ∂/∂z 1차 차분 행렬 (Ncells × Ncells).

        중심차분 사용 (경계는 단측 차분). Gx·m, Gz·m 둘 다 같은 셀 그리드에 살기 때문에
        Gxᵀ·Gz 형태의 교차항을 정확히 계산할 수 있다.
        """
        if self.use_blocks and self.inv_blocks is not None:
            ib = self.inv_blocks
            n = ib.n_blocks
            cx = np.array([blk['cx'] for blk in ib.blocks])
            cz = np.array([blk['cz'] for blk in ib.blocks])
            rows_x, cols_x, vals_x = [], [], []
            rows_z, cols_z, vals_z = [], [], []
            a3 = self.survey.a * 3.0
            for bi in range(n):
                dx_all = cx - cx[bi]
                dz_all = cz - cz[bi]
                # x 방향 인접: dz≈0, |dx|<a3
                mx = (np.abs(dz_all) < 1e-6) & (np.abs(dx_all) > 1e-6) & (np.abs(dx_all) < a3)
                idx_x = np.where(mx)[0]
                if len(idx_x):
                    for bj in idx_x:
                        sign = 1.0 if dx_all[bj] > 0 else -1.0
                        rows_x.append(bi); cols_x.append(bj); vals_x.append(sign / max(abs(dx_all[bj]) * 2, 1e-9))
                        rows_x.append(bi); cols_x.append(bi); vals_x.append(-sign / max(abs(dx_all[bj]) * 2, 1e-9))
                # z 방향 인접
                mz = (np.abs(dx_all) < 1e-6) & (np.abs(dz_all) > 1e-6) & (np.abs(dz_all) < a3)
                idx_z = np.where(mz)[0]
                if len(idx_z):
                    for bj in idx_z:
                        sign = 1.0 if dz_all[bj] > 0 else -1.0
                        rows_z.append(bi); cols_z.append(bj); vals_z.append(sign / max(abs(dz_all[bj]) * 2, 1e-9))
                        rows_z.append(bi); cols_z.append(bi); vals_z.append(-sign / max(abs(dz_all[bj]) * 2, 1e-9))
            Gx = sparse.csr_matrix((vals_x, (rows_x, cols_x)), shape=(n, n))
            Gz = sparse.csr_matrix((vals_z, (rows_z, cols_z)), shape=(n, n))
            return Gx, Gz

        m = self.mesh
        nc = m.n_cells
        rows_x, cols_x, vals_x = [], [], []
        rows_z, cols_z, vals_z = [], [], []
        for iz in range(m.ncz):
            for ix in range(m.ncx):
                ci = m.cidx(ix, iz)
                # ∂/∂x
                if 0 < ix < m.ncx - 1:
                    dx_full = m.x_cc[ix + 1] - m.x_cc[ix - 1]
                    rows_x.append(ci); cols_x.append(m.cidx(ix + 1, iz)); vals_x.append(1.0 / dx_full)
                    rows_x.append(ci); cols_x.append(m.cidx(ix - 1, iz)); vals_x.append(-1.0 / dx_full)
                elif ix == 0 and m.ncx > 1:
                    dx = m.x_cc[1] - m.x_cc[0]
                    rows_x.append(ci); cols_x.append(m.cidx(1, iz)); vals_x.append(1.0 / dx)
                    rows_x.append(ci); cols_x.append(m.cidx(0, iz)); vals_x.append(-1.0 / dx)
                elif ix == m.ncx - 1 and m.ncx > 1:
                    dx = m.x_cc[-1] - m.x_cc[-2]
                    rows_x.append(ci); cols_x.append(m.cidx(m.ncx - 1, iz)); vals_x.append(1.0 / dx)
                    rows_x.append(ci); cols_x.append(m.cidx(m.ncx - 2, iz)); vals_x.append(-1.0 / dx)
                # ∂/∂z
                if 0 < iz < m.ncz - 1:
                    dz_full = m.z_cc[iz + 1] - m.z_cc[iz - 1]
                    rows_z.append(ci); cols_z.append(m.cidx(ix, iz + 1)); vals_z.append(1.0 / dz_full)
                    rows_z.append(ci); cols_z.append(m.cidx(ix, iz - 1)); vals_z.append(-1.0 / dz_full)
                elif iz == 0 and m.ncz > 1:
                    dz = m.z_cc[1] - m.z_cc[0]
                    rows_z.append(ci); cols_z.append(m.cidx(ix, 1)); vals_z.append(1.0 / dz)
                    rows_z.append(ci); cols_z.append(m.cidx(ix, 0)); vals_z.append(-1.0 / dz)
                elif iz == m.ncz - 1 and m.ncz > 1:
                    dz = m.z_cc[-1] - m.z_cc[-2]
                    rows_z.append(ci); cols_z.append(m.cidx(ix, m.ncz - 1)); vals_z.append(1.0 / dz)
                    rows_z.append(ci); cols_z.append(m.cidx(ix, m.ncz - 2)); vals_z.append(-1.0 / dz)
        Gx = sparse.csr_matrix((vals_x, (rows_x, cols_x)), shape=(nc, nc))
        Gz = sparse.csr_matrix((vals_z, (rows_z, cols_z)), shape=(nc, nc))
        return Gx, Gz

    def _apply_anisotropic_rotation_proper(self):
        """**올바른** 회전 노름 기반 이방성 정규화 (참고: 기존 _apply_anisotropic_rotation의 교차항은
        대각 차분 Wxz² 근사라 부정확. 이 함수는 셀 중심 ∂/∂x, ∂/∂z를 사용해서
        D_s = cosθ·Gx + sinθ·Gz, D_t = -sinθ·Gx + cosθ·Gz 를 만들고
        w·D_sᵀD_s + (1/w)·D_tᵀD_t 로 정확한 회전 노름 구현.
        """
        theta = np.radians(self.dip_angle)
        c_t = float(np.cos(theta)); s_t = float(np.sin(theta))
        w = float(self.dip_weight)
        Gx, Gz = self._build_cell_center_gradients()
        # 회전 그라디언트
        Ds = (c_t * Gx) + (s_t * Gz)   # along feature
        Dt = (-s_t * Gx) + (c_t * Gz)  # across feature
        # 정확한 회전 노름
        WtW_aniso = w * (Ds.T @ Ds) + (1.0 / w) * (Dt.T @ Dt)
        # 작은 진폭 평활(Ws)는 유지
        self._WtW_aniso = (self.Ws.T @ self.Ws + WtW_aniso)

    def _compute_structure_tensor_dips(self, m_cur):
        """구조 텐서로부터 국소 경사각(라디안) 추출.

        Weickert (1998), Guenther et al. (2006) 기반.
        m_cur: 로그 비저항 모델 (n_cells,)
        반환: theta_field (ncz × ncx) - 각 셀의 국소 경사각 (라디안)
               0 = 수평, 양수 = 오른쪽 하향

        알고리즘:
          1. 모델을 2D 배열로 재구성 (ncz × ncx)
          2. 비저항 기울기 계산 (∂m/∂x, ∂m/∂z)
          3. 구조 텐서 요소 계산: Jxx=(∂m/∂x)², Jzz=(∂m/∂z)², Jxz=(∂m/∂x)(∂m/∂z)
          4. 가우시안 평활화로 국소 구조 일관성 확보
          5. 고유값 분해 → 큰 고유벡터 = 기울기 방향(= 지층 법선)
             지층 경사 = 법선으로부터 90° 회전
        """
        from scipy.ndimage import gaussian_filter

        m = self.mesh
        ncz, ncx = m.ncz, m.ncx

        # 셀 모드 전용 (블록 모드는 파라미터 수가 적어 구조 텐서 불필요)
        if self.use_blocks:
            return None

        # 2D 배열로 재구성 (log 비저항)
        lnrho = m_cur.reshape(ncz, ncx)

        # 중심 차분으로 기울기 계산 (경계는 단측 차분)
        dlnrho_dx = np.gradient(lnrho, m.x_cc, axis=1)  # ∂m/∂x
        dlnrho_dz = np.gradient(lnrho, m.z_cc, axis=0)  # ∂m/∂z

        # 구조 텐서 요소
        Jxx_raw = dlnrho_dx ** 2
        Jzz_raw = dlnrho_dz ** 2
        Jxz_raw = dlnrho_dx * dlnrho_dz

        # 가우시안 평활화 (σ 셀 단위)
        sig = self.st_smooth_sigma
        Jxx = gaussian_filter(Jxx_raw, sigma=sig)
        Jzz = gaussian_filter(Jzz_raw, sigma=sig)
        Jxz = gaussian_filter(Jxz_raw, sigma=sig)

        # 셀별 2×2 고유값 분해 → 우세 기울기 방향(= 지층 법선) 추출
        # 법선 방향에서 90° 시계방향 회전 → 경사 방향 (Weickert, 1998)
        trace = Jxx + Jzz
        det = Jxx * Jzz - Jxz ** 2
        disc = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0.0))
        lam1 = trace / 2 + disc   # 큰 고유값 (법선 방향)
        lam2 = trace / 2 - disc   # 작은 고유값

        # 큰 고유벡터 (법선 방향): (Jxz, lam1-Jxx) — 정규화 전
        vx = Jxz
        vz = lam1 - Jxx
        norm_v = np.sqrt(vx ** 2 + vz ** 2) + 1e-10

        # 부호 통일: vz >= 0 강제 (고유벡터 방향 모호성 해소)
        # z는 깊이(아래=양수). vz>=0이면 법선이 아래를 향함 → 경사 방향 일관성
        flip = vz < 0
        vx_c = np.where(flip, -vx, vx)
        vz_c = np.where(flip, -vz, vz)

        # 경사 방향 = 법선 벡터 90° 시계방향 회전: (n_x, n_z) → (n_z, -n_x)
        # 단, RESIS Pro 관례: theta = 수평으로부터 오른쪽 하향 각도
        # dip_direction = (vz_c, -vx_c) [경사 방향]
        # theta_dip = atan2(-vx_c, vz_c)
        #   수평층(vx_c=0, vz_c>0)  → atan2(0, vz_c) = 0° ✓
        #   20° 우하향(vx_c≈-sin20°, vz_c≈cos20°) → atan2(sin20°, cos20°) = 20° ✓
        #   수직층(vx_c>0, vz_c≈0) → atan2(-vx_c, 0) = ±90° ✓
        theta_field = np.arctan2(-vx_c / norm_v, vz_c / norm_v)

        # 구조 텐서 신뢰도 (coherence): 두 고유값 비율
        # coherence ≈ 1: 강한 단방향 구조 (경사 신뢰성 높음)
        # coherence ≈ 0: 등방적 구조 (경사 불확실 → 0°로 fallback)
        coherence = np.where(lam1 > 1e-10, (lam1 - lam2) / (lam1 + 1e-10), 0.0)
        coherence_thresh = 0.3
        scale = np.clip((coherence - coherence_thresh) / (1.0 - coherence_thresh), 0.0, 1.0)
        theta_field *= scale

        self._st_dip_field = theta_field.copy()   # 시각화/저장용
        self._st_confidence_field = scale.copy()
        return theta_field

    def _build_structure_tensor_WtW(self, theta_field,
                                    confidence_field=None,
                                    confidence_scalar=1.0):
        """국소 경사각 필드에서 공간 가변 이방성 WtW 행렬 구성.

        theta_field: (ncz × ncx) 배열 - 각 셀의 국소 경사각 (라디안)
        confidence_field: (ncz × ncx) 배열 또는 None. 주어지면 국소 coherence에
                         따라 이방성 강도를 L2(=1)와 최대값 사이에서 조절.
        confidence_scalar: 전역 OOD/ML 신뢰도 게이트(0~1).

        수식:
          Ds = diag(cos θ) Gx + diag(sin θ) Gz   (지층 따라가는 방향)
          Dt = diag(-sin θ) Gx + diag(cos θ) Gz  (지층 가로지르는 방향)
          w_i = 1 + g·C_i·(wmax-1)
          WtW_ST = Dsᵀdiag(w_i)Ds + Dtᵀdiag(1/w_i)Dt + Wsᵀ Ws

        장점: 지층 방향으로는 강한 평활(경계 아닌 곳), 법선 방향으로는 약한 페널티(경계 허용)
              단, OOD/low-coherence 영역에서는 자동으로 등방 L2에 가까워진다.
        """
        theta_vec = theta_field.ravel()   # (n_cells,)
        cos_t = np.cos(theta_vec)
        sin_t = np.sin(theta_vec)
        wmax = max(float(self.st_dip_weight), 1.0)
        g = float(np.clip(confidence_scalar, 0.0, 1.0))
        if confidence_field is None:
            conf_vec = np.ones_like(theta_vec)
        else:
            conf_vec = np.clip(np.asarray(confidence_field).ravel(), 0.0, 1.0)
            if conf_vec.size != theta_vec.size:
                conf_vec = np.ones_like(theta_vec)
        w_vec = 1.0 + g * conf_vec * (wmax - 1.0)

        Gx, Gz = self._build_cell_center_gradients()

        # 공간 가변 대각 행렬
        Dc = sparse.diags(cos_t, format='csr')
        Ds_mat = sparse.diags(sin_t, format='csr')
        Ds = Dc @ Gx + Ds_mat @ Gz          # along structure
        Dt = (-Ds_mat) @ Gx + Dc @ Gz       # across structure
        W_along = sparse.diags(w_vec, format='csr')
        W_cross = sparse.diags(1.0 / np.maximum(w_vec, 1e-8), format='csr')

        WtW_ST = (Ds.T @ W_along @ Ds + Dt.T @ W_cross @ Dt +
                  self.Ws.T @ self.Ws)
        # Woodbury 솔버 호환: dense 반환 (WtW_l2과 동일한 타입)
        return WtW_ST.toarray() if sparse.issparse(WtW_ST) else WtW_ST

    def _apply_anisotropic_rotation(self):
        """이방성 정규화: 경사 방향(dip_angle)으로 평활화 강도를 조절.

        교차항 구현 방식 분기:
          - self.aniso_proper=True (기본): 셀 중심 ∂/∂x, ∂/∂z 로 정확한 회전 노름
          - False: 기존 대각 차분 Wxz² 근사 (back-compat)
        """
        if getattr(self, 'aniso_proper', True):
            self._apply_anisotropic_rotation_proper()
            return
        theta = np.radians(self.dip_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        w = self.dip_weight

        WxTWx = self.Wx.T @ self.Wx
        WzTWz = self.Wz.T @ self.Wz

        a_xx = w * cos_t**2 + sin_t**2 / w
        a_zz = w * sin_t**2 + cos_t**2 / w
        a_xz = (w - 1.0 / w) * sin_t * cos_t  # 교차항 계수

        # ── 교차항: 대각선 차분 연산자 Wxz 구성 ──
        # (ix,iz)→(ix+1,iz+1) 대각 차분으로 ∂²m/∂x∂z 근사
        if self.use_blocks and self.inv_blocks is not None:
            ib = self.inv_blocks
            np_ = ib.n_blocks
            # 블록 중심 좌표 배열화 (벡터 연산)
            cx = np.array([blk['cx'] for blk in ib.blocks])
            cz = np.array([blk['cz'] for blk in ib.blocks])
            r, c, v = [], [], []; idx = 0
            a3 = self.survey.a * 3
            for bi in range(np_):
                dx_all = cx - cx[bi]
                dz_all = cz - cz[bi]
                # 대각 인접 블록만 (dx≠0, dz≠0, 거리 < 3a)
                mask = ((np.abs(dx_all) > 1e-6) & (np.abs(dz_all) > 1e-6) &
                        (np.abs(dx_all) < a3) & (np.abs(dz_all) < a3))
                for bj in np.where(mask)[0]:
                    if bj <= bi: continue  # 중복 방지
                    dl = np.sqrt(dx_all[bj]**2 + dz_all[bj]**2)
                    r.append(idx); c.append(bi); v.append(-1.0/dl)
                    r.append(idx); c.append(bj); v.append(1.0/dl)
                    idx += 1
            if idx > 0:
                Wxz = sparse.csr_matrix((v, (r, c)), shape=(idx, np_))
            else:
                Wxz = sparse.csr_matrix((1, np_))
            WxzTWxz = Wxz.T @ Wxz
        else:
            m = self.mesh
            nc = m.n_cells
            r, c, v = [], [], []; idx = 0
            for iz in range(m.ncz - 1):
                for ix in range(m.ncx - 1):
                    dx = m.x_cc[ix + 1] - m.x_cc[ix]
                    dz = m.z_cc[iz + 1] - m.z_cc[iz]
                    dl = np.sqrt(dx**2 + dz**2)
                    ci = m.cidx(ix, iz)
                    cj = m.cidx(ix + 1, iz + 1)
                    r.append(idx); c.append(ci); v.append(-1.0 / dl)
                    r.append(idx); c.append(cj); v.append(1.0 / dl)
                    idx += 1
            Wxz = sparse.csr_matrix((v, (r, c)), shape=(idx, nc))
            WxzTWxz = Wxz.T @ Wxz

        self._Wxz_base = Wxz  # MGS 재가중용 저장
        self._WtW_aniso = (self.Ws.T @ self.Ws +
                           a_xx * WxTWx + a_zz * WzTWz +
                           a_xz * WxzTWxz)

    def _precompute_cell_indices(self):
        """셀-노드 인덱스 사전 계산 (한 번만)"""
        m = self.mesh
        ix_arr, iz_arr = np.meshgrid(np.arange(m.ncx), np.arange(m.ncz))
        ix_f = ix_arr.ravel(); iz_f = iz_arr.ravel()
        self._n00 = iz_f * m.nx + ix_f
        self._n10 = self._n00 + 1
        self._n01 = self._n00 + m.nx
        self._n11 = self._n01 + 1
        self._cdx = m.dx[ix_f]
        self._cdz = m.dz[iz_f]
        self._carea = self._cdx * self._cdz

    def _sensitivity_kernel(self, phi_s, phi_r, ky):
        if not hasattr(self, '_n00'):
            self._precompute_cell_indices()
        n00, n10, n01, n11 = self._n00, self._n10, self._n01, self._n11
        dx, dz, area = self._cdx, self._cdz, self._carea
        dsdx = 0.5 * ((phi_s[n10] - phi_s[n00]) + (phi_s[n11] - phi_s[n01])) / dx
        drdx = 0.5 * ((phi_r[n10] - phi_r[n00]) + (phi_r[n11] - phi_r[n01])) / dx
        dsdz = 0.5 * ((phi_s[n01] - phi_s[n00]) + (phi_s[n11] - phi_s[n10])) / dz
        drdz = 0.5 * ((phi_r[n01] - phi_r[n00]) + (phi_r[n11] - phi_r[n10])) / dz
        savg = 0.25 * (phi_s[n00] + phi_s[n10] + phi_s[n01] + phi_s[n11])
        ravg = 0.25 * (phi_r[n00] + phi_r[n10] + phi_r[n01] + phi_r[n11])
        return area * (dsdx * drdx + dsdz * drdz + ky ** 2 * savg * ravg)

    def compute_jacobian(self, solver, survey, rho_a):
        if not hasattr(self, '_n00'):
            self._precompute_cell_indices()
        m = self.mesh
        nd = survey.n_data; nc = m.n_cells; n_ky = len(solver.ky)

        # 전극 → 인덱스 매핑
        elecs = sorted(solver.phi_ky_cache.keys(), key=lambda x: (x[0], x[1]))
        unique_sx = sorted(set(k[0] for k in solver.phi_ky_cache.keys()))
        sx_to_idx = {sx: i for i, sx in enumerate(unique_sx)}
        n_unique = len(unique_sx)

        # phi_all: (n_unique * n_ky, n_nodes) 2D 배열
        phi_all = np.zeros((n_unique * n_ky, m.n_nodes))
        elec_ky_map = np.zeros((n_unique, n_ky), dtype=np.int64)
        for (sx, iky), phi in solver.phi_ky_cache.items():
            idx = sx_to_idx[sx]
            row = idx * n_ky + iky
            phi_all[row, :] = phi
            elec_ky_map[idx, iky] = row

        # 측정별 전극 인덱스 배열
        c1_idx = np.array([sx_to_idx[ms['c1']] for ms in survey.measurements], dtype=np.int64)
        c2_idx = np.array([sx_to_idx[ms['c2']] for ms in survey.measurements], dtype=np.int64)
        p1_idx = np.array([sx_to_idx[ms['p1']] for ms in survey.measurements], dtype=np.int64)
        p2_idx = np.array([sx_to_idx[ms['p2']] for ms in survey.measurements], dtype=np.int64)
        K_arr = np.array([ms['K'] for ms in survey.measurements])

        J = np.zeros((nd, nc))
        _jacobian_core(J, nd, nc, n_ky,
                       c1_idx, c2_idx, p1_idx, p2_idx, K_arr, rho_a,
                       solver.ky, solver.ky_w, solver.sigma,
                       self._n00, self._n10, self._n01, self._n11,
                       self._cdx, self._cdz, self._carea,
                       phi_all, n_unique, elec_ky_map)
        return J

    def _find_alpha_lcurve(self, J, WdJ, wd, res, m_cur, m_ref, WtW, callback=None):
        """L-curve 기반 최적 정규화 파라미터 자동 선택

        여러 α로 Gauss-Newton 업데이트를 계산하여
        데이터 적합도(‖residual‖) vs 모델 평활도(‖Wm‖) 곡선의
        최대 곡률 지점을 찾음
        """
        # 셀+L2 (다중 스케일): α가 작아야 함 → 넓은 범위
        # 나머지: 표준 범위
        nd = len(res)
        ratio = self.n_params / max(nd, 1)
        if ratio > 10 and self.reg_type == 'L2':
            # 셀+L2: 다중 스케일 초기 모델에서 시작 → 더 넓은 α 탐색
            alphas = np.logspace(-4, 2, 16)
        else:
            alphas = np.logspace(-2, 3, 12)
        data_norms = []
        model_norms = []

        if callback:
            callback("L-curve: 최적 α 탐색 중...")

        for a_trial in alphas:
            lhs = WdJ.T @ WdJ + a_trial * WtW
            rhs_vec = WdJ.T @ (wd * res) - a_trial * WtW @ (m_cur - m_ref)
            try:
                dm = np.linalg.solve(lhs, rhs_vec)
            except np.linalg.LinAlgError:
                dm = np.linalg.lstsq(lhs, rhs_vec, rcond=None)[0]
            m_trial = m_cur + dm
            data_norms.append(np.linalg.norm(wd * res - np.diag(wd) @ J @ dm))
            model_norms.append(np.linalg.norm(WtW @ (m_trial - m_ref)))

        # log 공간에서 곡률 계산
        x = np.log10(np.array(data_norms) + 1e-30)
        y = np.log10(np.array(model_norms) + 1e-30)

        # 유한차분으로 곡률 κ 계산
        best_idx = len(alphas) // 2  # 기본값: 중간
        if len(x) >= 3:
            dx = np.diff(x); dy = np.diff(y)
            ddx = np.diff(dx); ddy = np.diff(dy)
            # 중앙 차분 곡률
            dx_c = 0.5 * (dx[:-1] + dx[1:])
            dy_c = 0.5 * (dy[:-1] + dy[1:])
            kappa = np.abs(dx_c * ddy - dy_c * ddx) / (dx_c**2 + dy_c**2 + 1e-30)**1.5
            best_idx = np.argmax(kappa) + 1  # +1: 중앙차분 오프셋

        best_alpha = alphas[best_idx]
        if callback:
            callback(f"L-curve: α = {best_alpha:.4f} 선택 (탐색 범위 {alphas[0]:.2e}~{alphas[-1]:.2e})")
        return best_alpha

    def _make_solver(self, rho_cells):
        """FDM 또는 FEM solver 생성 (FEM 시 FDM 격자 호환 래퍼 반환)"""
        if self.solver_type == 'FEM' and self.tri_mesh is not None:
            tm = self.tri_mesh; m = self.mesh
            # FDM 셀 비저항 → FEM 요소 비저항 매핑
            rho_elem = np.full(tm.n_elements, np.median(rho_cells))
            for ie in range(tm.n_elements):
                xc, zc = tm.elem_centers[ie]
                depth = tm.max_elev - zc if tm.has_topo else -zc
                ix = np.searchsorted(m.x_nodes, xc) - 1
                iz = np.searchsorted(m.z_nodes, max(depth, 0)) - 1
                ix = np.clip(ix, 0, m.ncx - 1)
                iz = np.clip(iz, 0, m.ncz - 1)
                rho_elem[ie] = rho_cells[m.cidx(ix, iz)]
            fem_solver = ForwardFEM(tm, rho_elem)
            # FDM 호환 래퍼: FEM 전위를 FDM 노드로 보간
            wrapper = _FEMtoFDMWrapper(fem_solver, m, tm)
            wrapper.sigma = 1.0 / rho_cells  # FDM 셀 기준 전도도
            return wrapper
        else:
            return ForwardSolver(self.mesh, rho_cells)

    def estimate_dip_from_model(self, m_cur, smooth_sigma=1.0):
        """구조 텐서 기반 우세 경사 방향 자동 추정 (Self-Estimating Anisotropy).

        영상처리의 structure tensor (Förstner, Big端n; Knutsson 1989) 방법을
        2D 비저항 모델 m에 적용:
            T = ⟨∇m · ∇mᵀ⟩
        T의 주(최대 고유값) 고유벡터는 gradient의 우세 방향 (= 경계 normal).
        경사 방향(feature direction)은 이에 수직.

        Parameters
        ----------
        m_cur : (n_cells,) log-비저항 벡터
        smooth_sigma : 텐서 성분에 적용할 가우시안 평활화 표준편차 (셀 단위)

        Returns
        -------
        dip_angle_deg : float
            우세 경사각 (수평=0°, 양수=오른쪽 하향). 코드의 dip_angle 관례와 동일.
        coherence : float
            이방성 강도 지표 ≈ sqrt(λ_max / λ_min). 1=등방, 클수록 강한 방향성.
            self.dip_weight 자동 설정용으로 사용 가능.
        """
        m2d = m_cur.reshape(self.mesh.ncz, self.mesh.ncx)
        # 셀 중심 기울기 (np.gradient는 중심차분 사용)
        gz, gx = np.gradient(m2d)
        # 구조 텐서 성분
        Jxx = gx * gx
        Jxz = gx * gz
        Jzz = gz * gz
        # 공간 평활화 — 노이즈를 줄이고 신뢰성 있는 방향 추정 (coherence 향상)
        if smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                Jxx = gaussian_filter(Jxx, smooth_sigma)
                Jxz = gaussian_filter(Jxz, smooth_sigma)
                Jzz = gaussian_filter(Jzz, smooth_sigma)
            except Exception:
                pass
        # 전역 텐서 (셀 가중 평균)
        Txx = float(np.mean(Jxx))
        Tzz = float(np.mean(Jzz))
        Txz = float(np.mean(Jxz))
        trace = Txx + Tzz
        det = Txx * Tzz - Txz * Txz
        disc = max(trace * trace * 0.25 - det, 0.0)
        lam_max = trace * 0.5 + np.sqrt(disc)
        lam_min = max(trace * 0.5 - np.sqrt(disc), 1e-30)
        # 주 고유벡터 방향 (gradient 우세 방향, = 경계 normal)
        # closed-form: θ = ½·atan2(2·Txz, Txx − Tzz)
        if abs(Txx - Tzz) < 1e-30 and abs(Txz) < 1e-30:
            theta_grad_rad = 0.0
        else:
            theta_grad_rad = 0.5 * np.arctan2(2.0 * Txz, Txx - Tzz)
        # 경계 normal 방향에 수직인 게 feature(경사) 방향
        feature_rad = theta_grad_rad + np.pi * 0.5
        feature_deg = np.degrees(feature_rad)
        # [-90, 90]으로 정규화
        while feature_deg > 90:
            feature_deg -= 180
        while feature_deg < -90:
            feature_deg += 180
        coherence = float(np.sqrt(lam_max / lam_min))
        return float(feature_deg), coherence

    def _compute_mgs_WtW(self, m_cur, m_ref):
        """MGS (Minimum Gradient Support) 정규화 행렬 계산.

        기울기가 큰 곳(경계)은 페널티를 줄여 날카로운 경계 허용.
        β는 모델 기울기의 절대값 분포 기반으로 자동 설정.
        """
        dm = m_cur - m_ref

        def reweight_matrix(W_orig):
            if W_orig.shape[0] == 0:
                return W_orig
            Wdm = (W_orig @ dm)  # 각 차분 값
            abs_wdm = np.abs(Wdm)
            # β: 10번째 백분위수, 단 중앙값의 10% 이상을 보장
            # (블록→셀 보간 시 대부분 0이면 percentile(10)≈0 → 이진 가중치 방지)
            p10 = np.percentile(abs_wdm, 10)
            med_floor = np.median(abs_wdm) * 0.1
            beta = max(p10, med_floor, 1e-6) * self.mgs_beta
            beta2 = beta ** 2
            mgs_w = beta2 / (Wdm ** 2 + beta2)
            # 최소 가중치 클리핑 (완전히 0이 되지 않도록)
            mgs_w = np.clip(mgs_w, self.mgs_weight_clip, 1.0)
            R = sparse.diags(np.sqrt(mgs_w))
            return R @ W_orig

        Ws_r = reweight_matrix(self.Ws)
        Wx_r = reweight_matrix(self.Wx)
        Wz_r = reweight_matrix(self.Wz)

        # 이방성 적용 (교차항 포함)
        if abs(self.dip_angle) > 0.1 and self.dip_weight > 1.0:
            theta = np.radians(self.dip_angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            w = self.dip_weight
            a_xx = w * cos_t**2 + sin_t**2 / w
            a_zz = w * sin_t**2 + cos_t**2 / w
            a_xz = (w - 1.0 / w) * sin_t * cos_t
            # 대각선 차분도 MGS 재가중
            if hasattr(self, '_Wxz_base'):
                Wxz_r = reweight_matrix(self._Wxz_base)
                WxzTWxz_r = Wxz_r.T @ Wxz_r
            else:
                WxzTWxz_r = sparse.csr_matrix((Wx_r.shape[1], Wx_r.shape[1]))
            return (Ws_r.T @ Ws_r +
                    a_xx * (Wx_r.T @ Wx_r) +
                    a_zz * (Wz_r.T @ Wz_r) +
                    a_xz * WxzTWxz_r).toarray()

        return (Ws_r.T @ Ws_r + Wx_r.T @ Wx_r + Wz_r.T @ Wz_r).toarray()

    def run(self, d_obs, callback=None, auto_alpha=False, robust=False):
        m = self.mesh
        np_ = self.n_params
        nd = len(d_obs)
        ib = self.inv_blocks

        m_cur = np.full(np_, np.log(self.rho_ref)); m_ref = m_cur.copy()
        has_initial_model = self.initial_model is not None
        if has_initial_model:
            init_arr = np.asarray(self.initial_model, dtype=float).ravel()
            if init_arr.size == np_:
                m_cur = np.log(np.clip(init_arr, 0.1, 1e5))
                m_ref = m_cur.copy()
        if self.reference_model is not None:
            ref_arr = np.asarray(self.reference_model, dtype=float).ravel()
            if ref_arr.size == np_:
                m_ref = np.log(np.clip(ref_arr, 0.1, 1e5))
        d_log = np.log(np.maximum(np.abs(d_obs), 1e-10))

        # ── 통계적 데이터 오차 모델 (LaBrecque et al., 1996) ──
        d_abs = np.maximum(np.abs(d_obs), 1e-10)
        self.data_errors = np.sqrt(
            self.pct_error**2 + (self.noise_floor / d_abs)**2)
        wd_base = 1.0 / self.data_errors
        wd = wd_base.copy()

        # 수렴 이력 초기화
        self.convergence = {
            'rms': [], 'chi2': [], 'roughness': [],
            'alpha': [], 'phi_d': [], 'phi_m': []
        }

        # L2: 고정 WtW, MGS: 매 반복 갱신
        if hasattr(self, '_WtW_aniso'):
            WtW_l2 = self._WtW_aniso.toarray() if sparse.issparse(self._WtW_aniso) else self._WtW_aniso
        else:
            WtW_l2 = (self.Ws.T @ self.Ws + self.Wx.T @ self.Wx + self.Wz.T @ self.Wz).toarray()
        WtW = WtW_l2.copy(); history = []; d_calc = d_obs.copy()

        alpha = self.alpha; alpha_selected = False
        use_fem = (self.solver_type == 'FEM' and self.tri_mesh is not None)
        is_cell_mode = not self.use_blocks

        aniso_str = ""
        if abs(self.dip_angle) > 0.1 and self.dip_weight > 1.0:
            aniso_str = f", 이방성 {self.dip_angle}°×{self.dip_weight}"

        if callback:
            mode = "블록" if ib else "셀"
            reg = self.reg_type
            callback(f"{mode} 역산 ({self.n_params}개), 정규화: {reg}{aniso_str}")
            if is_cell_mode:
                callback(f"  파라미터/데이터 비율: {np_/nd:.0f}:1 (과소결정)")
            if reg == 'MGS':
                callback(f"  MGS β = {self.mgs_beta}")

        # ── 다중 스케일 초기화 (셀 모드) ──
        # Loke & Barker (1996), Günther et al. (2006)
        # 셀 모드에서는 블록 역산을 먼저 수행하여 초기 모델 생성.
        # L2: Phase 1 결과를 초기/참조 모델로
        # MGS: Phase 1 결과를 초기 모델로, 참조는 균일 (구조 감지용)
        do_phase1 = is_cell_mode and not (has_initial_model and self.skip_phase1_if_initial)
        if do_phase1:
            if callback:
                callback("── Phase 1: 블록 사전 역산 ──")
            pre_inv = Inversion2D(
                self.survey, m, rho_ref=self.rho_ref,
                alpha=self.alpha, alpha_s=self.alpha_s,
                alpha_x=self.alpha_x, alpha_z=self.alpha_z,
                max_iter=5, tol=0.05, solver_type='FDM',
                use_blocks=True, reg_type='L2',
                noise_floor=self.noise_floor, pct_error=self.pct_error,
                target_chi2=self.target_chi2, cooling_factor=0.7)
            pre_rho, pre_hist, pre_dcalc = pre_inv.run(
                d_obs, callback=callback, auto_alpha=auto_alpha, robust=robust)
            # 블록 결과를 셀 초기 모델로 사용
            m_cur = np.log(np.clip(pre_rho, 0.1, 1e5))
            if self.reg_type == 'MGS':
                # MGS: 균일 참조 → dm에 Phase 1 구조 반영 → 경계 감지
                m_ref = np.full(np_, np.log(self.rho_ref))
            else:
                # L2: Phase 1 결과 참조 → 블록 해 근처에서 세밀화
                m_ref = m_cur.copy()
            # Phase 1의 α를 Phase 2 시작점으로
            if hasattr(pre_inv, '_final_stats'):
                alpha = pre_inv._final_stats.get('alpha_final', self.alpha)
                alpha_selected = True  # 이미 L-curve 탐색 완료
            if callback:
                pre_rms = pre_hist[-1] if pre_hist else 0
                callback(f"── Phase 1 완료: RMS={pre_rms:.4f} → 셀 세밀 역산 시작 ──")

            # ── 구조 텐서 초기화 (Phase 1 결과로 초기 경사각 추정) ──
            if self.use_structure_tensor and not self.use_blocks:
                theta_field = self._compute_structure_tensor_dips(m_cur)
                if theta_field is not None:
                    WtW_l2 = self._build_structure_tensor_WtW(theta_field)
                    WtW = WtW_l2.copy()
                    dip_deg_est = float(np.degrees(np.median(np.abs(theta_field))))
                    if callback:
                        callback(f"  구조 텐서: 국소 경사각 추정 완료 (중앙값 {dip_deg_est:.1f}°)")
        elif is_cell_mode and has_initial_model and callback:
            callback("── Phase 1 생략: 제공된 초기 모델에서 셀 역산 시작 ──")

        # MGS 전환 시점
        mgs_start = 2 if self.reg_type == 'MGS' else self.max_iter + 1
        mgs_active = False
        mgs_rms_at_switch = None
        m_backup = None  # MGS 전환 전 모델 백업

        # ── Occam 최적 모델 추적 (χ²가 target에 가장 가까운 평활 모델) ──
        # 과적합(χ²<<target) 방지: 가열 중 오버슈트 대비 best 모델 보관
        m_best = None
        chi2_best_dist = np.inf   # |chi2 - target| 최소
        chi2_best = None
        rms_best = None

        for it in range(self.max_iter):
            # MGS 전환/복귀 관리
            if self.reg_type == 'MGS' and it >= mgs_start:
                if not mgs_active:
                    # 첫 전환: 백업 저장
                    m_backup = m_cur.copy()
                    alpha_backup = alpha
                    mgs_rms_at_switch = history[-1] if history else 1.0
                    mgs_active = True
                    mgs_iter_count = 0
                    WtW_mgs = self._compute_mgs_WtW(m_cur, m_ref)
                    # 계측: MGS vs L2 정규화 강도 비율 (Frobenius norm)
                    if self.record_condition:
                        l2_arr = WtW_l2.toarray() if sparse.issparse(WtW_l2) else WtW_l2
                        mgs_arr = WtW_mgs.toarray() if sparse.issparse(WtW_mgs) else WtW_mgs
                        self._mgs_l2_ratio = float(np.linalg.norm(l2_arr, 'fro') /
                                                   max(np.linalg.norm(mgs_arr, 'fro'), 1e-30))
                    # 점진적 블렌딩: 첫 MGS 반복은 L2 50% + MGS 50% (ablation off → 즉시 100%)
                    if self.mgs_blending:
                        WtW = 0.5 * WtW_l2 + 0.5 * WtW_mgs
                    else:
                        WtW = WtW_mgs
                    # MGS α 축소
                    if is_cell_mode:
                        alpha = alpha * self.mgs_alpha_factor_cell
                    else:
                        alpha = alpha * self.mgs_alpha_factor_block
                    if callback:
                        callback(f"  → MGS 전환 (반복 {it+1}), α: {alpha_backup:.4f} → {alpha:.4f} (블렌딩 50%)")
                elif mgs_active:
                    mgs_iter_count += 1
                    # 발산 감지: RMS가 전환 시점보다 N배 이상 악화
                    if history and history[-1] > mgs_rms_at_switch * self.mgs_divergence_threshold:
                        mgs_active = False
                        alpha = alpha_backup
                        WtW = WtW_l2.copy()
                        m_cur = m_backup.copy()
                        if callback:
                            callback(f"  ← MGS 발산 감지, L2로 복귀 (α={alpha:.4f})")
                    else:
                        WtW_mgs = self._compute_mgs_WtW(m_cur, m_ref)
                        # 2번째 MGS 반복부터 완전 MGS (블렌딩 해제)
                        if mgs_iter_count >= 2 or not self.mgs_blending:
                            WtW = WtW_mgs
                        else:
                            WtW = 0.3 * WtW_l2 + 0.7 * WtW_mgs

            # ── 구조 텐서 WtW 갱신 (셀 모드, L2, st_update_interval마다) ──
            # MGS가 활성이면 MGS가 WtW를 관리하므로 구조 텐서는 건너뜀
            if (self.use_structure_tensor and is_cell_mode and
                    not mgs_active and self.reg_type == 'L2'):
                # Phase 1 이후 첫 반복(it=0) 또는 st_update_interval 마다 갱신
                if it == 0 or (it % self.st_update_interval == 0):
                    theta_field = self._compute_structure_tensor_dips(m_cur)
                    if theta_field is not None:
                        WtW_l2 = self._build_structure_tensor_WtW(theta_field)
                        WtW = WtW_l2.copy()
                        if it > 0 and callback:
                            dip_med = float(np.degrees(np.median(np.abs(theta_field))))
                            callback(f"  구조 텐서 갱신 (반복 {it+1}): 중앙 경사각 {dip_med:.1f}°")

            # 블록 → 셀 비저항 변환
            if ib:
                rho_cells = ib.blocks_to_cells(np.exp(m_cur))
            else:
                rho_cells = np.exp(m_cur)

            # 전방 모델링
            if use_fem:
                solver_fwd = self._make_solver(rho_cells)
                if callback: callback(f"반복 {it+1}/{self.max_iter}: FEM 전방...")
                d_calc = solver_fwd.compute_data(self.survey)
            else:
                solver_fwd = ForwardSolver(m, rho_cells)
                if callback: callback(f"반복 {it+1}/{self.max_iter}: FDM 전방...")
                d_calc = solver_fwd.compute_data(self.survey)

            dc_log = np.log(np.maximum(np.abs(d_calc), 1e-10))
            res = d_log - dc_log; rms = np.sqrt(np.mean(res ** 2))

            # ── chi-squared 적합도 (Constable et al., 1987) ──
            weighted_res = wd * res
            phi_d = np.sum(weighted_res ** 2)       # 데이터 목적함수
            chi2 = phi_d / nd                        # chi²/N (목표: target_chi2)
            # 모델 거칠기 (정규화 목적함수)
            dm_ref = m_cur - m_ref
            if sparse.issparse(WtW):
                phi_m = float(dm_ref @ WtW @ dm_ref)
            else:
                phi_m = float(dm_ref @ (WtW @ dm_ref))
            roughness = phi_m

            if robust and it > 0:
                abs_res = np.abs(res)
                huber_c = 1.5 * np.median(abs_res)
                huber_w = np.where(abs_res <= huber_c, 1.0, huber_c / (abs_res + 1e-10))
                wd = wd_base * huber_w
                n_down = np.sum(huber_w < 0.9)
                if callback and n_down > 0:
                    callback(f"  Robust: {n_down}개 가중치 하향")

            history.append(rms)
            self.convergence['rms'].append(rms)
            self.convergence['chi2'].append(chi2)
            self.convergence['roughness'].append(roughness)
            self.convergence['alpha'].append(alpha)
            self.convergence['phi_d'].append(phi_d)
            self.convergence['phi_m'].append(phi_m)

            if callback:
                callback(f"반복 {it+1}: RMS={rms:.4f}, χ²/N={chi2:.3f}, "
                         f"α={alpha:.4e}, φ_d={phi_d:.1f}, φ_m={roughness:.2f}")

            # ── Occam 최적 모델 추적 ──
            # target에 가장 가까우면서 과적합(χ²<target)을 우선 보관하지 않도록
            # χ² ≥ target*0.9 인 모델 중 target에 가장 가까운 것을 best로 저장
            dist = abs(chi2 - self.target_chi2)
            if chi2 >= self.target_chi2 * 0.9 and dist < chi2_best_dist:
                chi2_best_dist = dist
                m_best = m_cur.copy()
                chi2_best = chi2
                rms_best = rms

            # ── 수렴 판정 (chi-squared 기반, 진짜 Occam's criterion) ──
            # 최소 2회 반복 보장 (균일 초기모델의 우연한 적합 방지)
            # 핵심: χ²<target*0.9 (과적합)이면 수렴하지 않고 가열(α↑)로 평활화
            min_iter = max(2, mgs_start + 3) if self.reg_type == 'MGS' else 2
            converged = False
            if it >= min_iter:
                if self.target_chi2 * 0.9 <= chi2 <= self.target_chi2 * 1.1:
                    # 적합 밴드 도달 → 수렴
                    converged = True
                    if callback:
                        callback(f"  ✓ χ²/N={chi2:.3f} (목표 {self.target_chi2:.2f} 밴드), 수렴")
                elif chi2 < self.target_chi2 * 0.9:
                    # 과적합 상태 → 가열로 χ²를 target까지 끌어올림 (수렴 보류)
                    # 단, 가열해도 더 나아지지 않으면(이미 best가 있고 RMS 정체) 종료
                    if callback:
                        callback(f"  ⚠ χ²/N={chi2:.3f} < 목표 → 과적합, 가열 계속")
                elif rms < self.tol and chi2 <= self.target_chi2 * 1.1:
                    converged = True
            if converged:
                break

            # 자코비안 (FDM 기반)
            if callback: callback(f"반복 {it+1}: 자코비안...")
            solver_fdm = ForwardSolver(m, rho_cells)
            solver_fdm.compute_data(self.survey)
            J_cells = self.compute_jacobian(solver_fdm, self.survey, d_calc)

            # 블록 자코비안으로 변환
            J = ib.cells_to_blocks_jacobian(J_cells) if ib else J_cells
            self._last_J = J  # 감도/분해능 분석용 저장
            WdJ = np.diag(wd) @ J

            if auto_alpha and not alpha_selected:
                alpha = self._find_alpha_lcurve(
                    J, WdJ, wd, res, m_cur, m_ref, WtW, callback)
                alpha_selected = True

            # ── Occam's cooling (Constable et al., 1987) ──
            # chi² > target: α 감소 (더 적합) / chi² < target: α 증가 (더 평활)
            if it > 0 and alpha_selected:
                if chi2 > self.target_chi2 * 1.5:
                    cool_f = self.cooling_factor
                    # 셀+L2: 다중 스케일 후 빠르게 수렴 유도
                    if is_cell_mode and self.reg_type == 'L2':
                        cool_f = cool_f ** 1.5  # 0.7→0.585
                    # 셀+MGS: Woodbury 안정성 위해 보수적 냉각
                    elif is_cell_mode and mgs_active:
                        cool_f = 0.85  # MGS에서 α를 너무 줄이면 aR 조건수 악화
                    new_alpha = max(alpha * cool_f, self.min_alpha)
                    if new_alpha != alpha:
                        if callback:
                            callback(f"  Occam 냉각: α {alpha:.4e} → {new_alpha:.4e}")
                        alpha = new_alpha
                elif chi2 < self.target_chi2 * 0.8:
                    new_alpha = alpha / self.cooling_factor
                    if callback:
                        callback(f"  Occam 가열: α {alpha:.4e} → {new_alpha:.4e}")
                    alpha = new_alpha

            dm_ref = m_cur - m_ref
            c_vec = WdJ.T @ (wd * res) - alpha * WtW @ dm_ref

            if is_cell_mode and np_ > nd * 5:
                # ── 데이터 공간 GN 솔버 (Woodbury Identity) ──
                # Siripunvaraporn & Egbert (2000), Tarantola & Valette (1982)
                #
                # 모델 공간 (np×np):  (J^T D J + αR) dm = c
                # 데이터 공간 (nd×nd): dm = v1 - V2 @ solve(D^{-1}+G, J·v1)
                #
                # 여기서:
                #   v1 = (αR)^{-1} c       (np-벡터, 1회 풀기)
                #   V2 = (αR)^{-1} J^T     (np×nd, nd회 풀기)
                #   G = J V2               (nd×nd)
                #   D^{-1} = diag(1/wd²)
                #
                # 장점: 수학적으로 동일하지만 nd×nd 시스템만 풀면 됨
                #   → null-space 수치 오염 없음, 조건수 대폭 개선
                aR = alpha * WtW
                # MGS 안정화: 대각 정규화 추가 (Tikhonov ridge)
                eps_used = 0.0
                diag_pre_max = np.nan; diag_pre_min = np.nan
                diag_post_max = np.nan; diag_post_min = np.nan
                if mgs_active:
                    diag_aR = np.abs(np.diag(aR.toarray() if sparse.issparse(aR) else aR))
                    if self.record_condition:
                        diag_pre_max = float(np.max(diag_aR))
                        diag_pre_min = float(np.min(diag_aR) + 1e-30)
                    diag_mean = np.mean(diag_aR) + 1e-10
                    eps_used = diag_mean * self.ridge_ratio
                    aR = aR + eps_used * np.eye(np_)
                    if self.record_condition:
                        diag_aR_post = np.abs(np.diag(aR.toarray() if sparse.issparse(aR) else aR))
                        diag_post_max = float(np.max(diag_aR_post))
                        diag_post_min = float(np.min(diag_aR_post) + 1e-30)
                # c와 J^T를 합쳐서 한 번에 풀기 (LU 재사용)
                combined_rhs = np.column_stack([c_vec.reshape(-1, 1), J.T])
                try:
                    combined_sol = np.linalg.solve(aR, combined_rhs)
                except np.linalg.LinAlgError:
                    combined_sol = np.linalg.lstsq(aR, combined_rhs, rcond=None)[0]
                v1 = combined_sol[:, 0]      # (αR)^{-1} c
                V2 = combined_sol[:, 1:]     # (αR)^{-1} J^T, shape: (np, nd)
                # 데이터 공간 Gram 행렬 (nd × nd)
                G = J @ V2
                # 데이터 공간 시스템 풀기 (nd × nd, 잘 조건화됨)
                D_inv = np.diag(1.0 / (wd ** 2 + 1e-20))
                lhs_data = D_inv + G
                rhs_data = J @ v1
                # 계측: 데이터 공간 LHS 조건수 (100×100, 빠름)
                if self.record_condition and mgs_active:
                    try:
                        cond_data = float(np.linalg.cond(lhs_data))
                    except Exception:
                        cond_data = np.inf
                    self.cond_history.append({
                        'iter': it + 1,
                        'alpha': float(alpha),
                        'eps_ridge': float(eps_used),
                        'diag_ratio_pre': float(diag_pre_max / max(diag_pre_min, 1e-30)),
                        'diag_ratio_post': float(diag_post_max / max(diag_post_min, 1e-30)),
                        'cond_data_lhs': float(cond_data),
                        'rms': float(history[-1]) if history else np.nan,
                    })
                q = np.linalg.solve(lhs_data, rhs_data)
                # 모델 업데이트 복원 (데이터 지원 부분공간만)
                dm = v1 - V2 @ q
                if callback:
                    callback(f"  데이터공간 GN: {np_}→{nd} ({np_/nd:.0f}:1 축소)")
            else:
                # 블록 모드: 표준 모델 공간 직접 풀기
                lhs = WdJ.T @ WdJ + alpha * WtW
                try: dm = np.linalg.solve(lhs, c_vec)
                except np.linalg.LinAlgError:
                    dm = np.linalg.lstsq(lhs, c_vec, rcond=None)[0]

            # ── 모델 업데이트 제한 (안정화) ──
            dm_max = 3.0 if is_cell_mode else 5.0
            dm_norm = np.max(np.abs(dm))
            if dm_norm > dm_max:
                dm = dm * (dm_max / dm_norm)
                if callback:
                    callback(f"  업데이트 제한: {dm_norm:.2f} → {dm_max:.1f}")

            # ── 스텝 평가 헬퍼 (forward solve → phi_d, chi2, rms) ──
            def _eval_step(s):
                m_t = m_cur + s * dm
                rho_t_block = np.clip(np.exp(m_t), 0.1, 1e5)
                rho_t_cells = ib.blocks_to_cells(rho_t_block) if ib else rho_t_block
                if use_fem:
                    sol_t = self._make_solver(rho_t_cells)
                else:
                    sol_t = ForwardSolver(m, rho_t_cells)
                d_t = sol_t.compute_data(self.survey)
                dt_l = np.log(np.maximum(np.abs(d_t), 1e-10))
                res_t = d_log - dt_l
                phi_t = np.sum((wd * res_t) ** 2)
                return phi_t, phi_t / nd, np.sqrt(np.mean(res_t ** 2))

            # ── Occam 스텝 제어 + Armijo line search ──
            # χ²가 target 위에 있을 때 full step이 target 아래로 오버슈트하면
            # 이분탐색으로 χ²≈target에 착지 (과적합 방지, Constable et al. 1987)
            armijo_c = 1e-4
            grad_dot_dm = np.dot(WdJ.T @ (wd * res), dm)
            target = self.target_chi2
            phi_full, chi2_full, rms_full = _eval_step(1.0)

            if chi2 > target and chi2_full < target * 0.95:
                # 오버슈트 감지 → χ²≈target 스텝 이분탐색
                lo, hi = 0.0, 1.0   # χ²(lo)=chi2(높음), χ²(hi)=chi2_full(낮음)
                step = 1.0; rms_try = rms_full; chi2_try = chi2_full
                for _ in range(7):
                    mid = 0.5 * (lo + hi)
                    phi_m_, chi2_m_, rms_m_ = _eval_step(mid)
                    if chi2_m_ > target:
                        lo = mid        # 아직 적합 부족 → 스텝 키움
                    else:
                        hi = mid         # 오버슈트 → 스텝 줄임
                    # target 이상에서 가장 가까운 스텝 채택
                    if chi2_m_ >= target * 0.95:
                        step = mid; rms_try = rms_m_; chi2_try = chi2_m_
                if callback:
                    callback(f"  Occam 스텝 제어: χ²/N {chi2:.3f}→{chi2_try:.3f} "
                             f"(step={step:.3f}, 과적합 방지)")
            else:
                # 표준 Armijo backtracking
                step = 1.0; rms_try = rms_full; chi2_try = chi2_full
                phi_d_try = phi_full
                for ls_iter in range(8):
                    if phi_d_try < phi_d - armijo_c * step * abs(grad_dot_dm):
                        break
                    if rms_try < rms:
                        break
                    step *= 0.5
                    phi_d_try, chi2_try, rms_try = _eval_step(step)

            m_cur = np.log(np.clip(np.exp(m_cur + step * dm), 0.1, 1e5))
            if callback:
                callback(f"반복 {it+1} 완료: RMS {rms:.4f}→{rms_try:.4f}, "
                         f"χ²/N {chi2:.3f}→{chi2_try:.3f}, step={step:.3f}")

        # ── Occam 최적 모델 선택 ──
        # best 모델(χ²가 target에 가장 가까운 평활 해)이 있으면 그것을 반환.
        # 과적합 방지: 마지막 m_cur가 과적합 상태일 수 있으므로 best 우선.
        final_chi2 = self.convergence['chi2'][-1] if self.convergence['chi2'] else 0
        final_rms = history[-1] if history else 0
        if m_best is not None and chi2_best is not None:
            # best가 마지막보다 target에 더 가까우면 채택
            last_dist = abs(final_chi2 - self.target_chi2)
            if chi2_best_dist < last_dist:
                if callback:
                    callback(f"  → Occam 최적 모델 채택: χ²/N {final_chi2:.3f} → "
                             f"{chi2_best:.3f} (과적합 방지)")
                m_cur = m_best
                final_chi2 = chi2_best
                final_rms = rms_best if rms_best is not None else final_rms

        # 최종 결과: 블록 → 셀 변환하여 반환
        rho_block = np.exp(m_cur)
        if ib:
            rho_result = ib.blocks_to_cells(rho_block)
            self._last_block_rho = rho_block  # 블록 단위 결과 저장
        else:
            rho_result = rho_block
            self._last_block_rho = None

        # 최종 적합도 통계 저장
        self._final_stats = {
            'n_iter': len(history),
            'rms_final': final_rms,
            'chi2_final': final_chi2,
            'alpha_final': alpha,
            'n_data': nd, 'n_params': np_,
        }
        return rho_result, history, d_calc

    # ── 학술 분석 메서드 ──

    def compute_sensitivity(self):
        """누적 감도 (Cummings & Zohdy, 1997; Friedel, 2003).

        S_j = Σ_i |J_ij| : 각 모델 파라미터에 대한 데이터의 총 민감도.
        정규화된 감도는 탐사 설계의 공간 분해 능력을 정량화한다.

        반환: 정규화된 감도 배열 (0~1), 블록 또는 셀 기준
        """
        J = self._last_J
        if J is None:
            raise RuntimeError("역산을 먼저 실행하세요 (자코비안 미저장)")
        sens = np.sum(np.abs(J), axis=0)
        # 로그 정규화 (Friedel, 2003): 동적 범위 압축
        sens_log = np.log10(np.maximum(sens, 1e-30))
        s_min, s_max = sens_log.min(), sens_log.max()
        if s_max - s_min < 1e-10:
            return np.ones_like(sens)
        return (sens_log - s_min) / (s_max - s_min)

    def compute_resolution_diagonal(self):
        """모델 분해능 행렬의 대각 성분 (Menke, 2012).

        R = (J^T W_d^T W_d J + α W^T W)^{-1} J^T W_d^T W_d J
        diag(R) → 1이면 완전 분해, 0이면 분해 불가.

        반환: 분해능 대각 배열, 블록 또는 셀 기준
        """
        J = self._last_J
        if J is None:
            raise RuntimeError("역산을 먼저 실행하세요 (자코비안 미저장)")
        alpha = self.convergence['alpha'][-1] if self.convergence['alpha'] else self.alpha
        WtW = self._WtW_aniso if hasattr(self, '_WtW_aniso') else \
              (self.Ws.T @ self.Ws + self.Wx.T @ self.Wx + self.Wz.T @ self.Wz)
        if sparse.issparse(WtW):
            WtW = WtW.toarray()

        JtJ = J.T @ J
        lhs = JtJ + alpha * WtW
        try:
            R = np.linalg.solve(lhs, JtJ)
        except np.linalg.LinAlgError:
            R = np.linalg.lstsq(lhs, JtJ, rcond=None)[0]
        return np.clip(np.diag(R), 0, 1)

    def compute_model_covariance_diagonal(self):
        """모델 공분산 행렬의 대각 성분 (후방 불확실성 추정).

        C_m = (J^T W_d^T W_d J + α W^T W)^{-1}
        sqrt(diag(C_m)) → 로그 공간 표준편차, 비저항 불확실성 인자로 변환 가능.

        반환: log 공간 표준편차 배열
        """
        J = self._last_J
        if J is None:
            raise RuntimeError("역산을 먼저 실행하세요")
        alpha = self.convergence['alpha'][-1] if self.convergence['alpha'] else self.alpha
        wd = 1.0 / self.data_errors if hasattr(self, 'data_errors') else np.ones(J.shape[0])
        WtW = self._WtW_aniso if hasattr(self, '_WtW_aniso') else \
              (self.Ws.T @ self.Ws + self.Wx.T @ self.Wx + self.Wz.T @ self.Wz)
        if sparse.issparse(WtW):
            WtW = WtW.toarray()

        WdJ = np.diag(wd) @ J
        lhs = WdJ.T @ WdJ + alpha * WtW
        try:
            C_m = np.linalg.inv(lhs)
        except np.linalg.LinAlgError:
            C_m = np.linalg.pinv(lhs)
        return np.sqrt(np.maximum(np.diag(C_m), 0))

    def export_results(self, filepath, rho_result, d_obs, d_calc):
        """역산 결과를 학술 표준 형식으로 내보내기.

        3개 파일 생성:
          {name}_model.xyz   - 모델 파라미터 (x, z, rho, sensitivity, resolution)
          {name}_data.xyz    - 데이터 적합도 (x, z, obs, calc, misfit%)
          {name}_convergence.csv - 수렴 이력 (iteration, rms, chi2, alpha, phi_d, phi_m)
        """
        base = os.path.splitext(filepath)[0]

        # ── 모델 파라미터 파일 ──
        try:
            sens = self.compute_sensitivity()
            res_diag = self.compute_resolution_diagonal()
        except RuntimeError:
            sens = np.zeros(self.n_params)
            res_diag = np.zeros(self.n_params)

        ib = self.inv_blocks
        with open(base + '_model.xyz', 'w') as f:
            f.write("# RESIS Pro Inversion Result\n")
            f.write(f"# Date: {__import__('datetime').datetime.now().isoformat()}\n")
            stats = getattr(self, '_final_stats', {})
            f.write(f"# Iterations: {stats.get('n_iter', '?')}, "
                    f"Final RMS: {stats.get('rms_final', 0):.5f}, "
                    f"Chi2/N: {stats.get('chi2_final', 0):.4f}\n")
            f.write(f"# N_data: {stats.get('n_data', '?')}, "
                    f"N_params: {stats.get('n_params', '?')}, "
                    f"Alpha: {stats.get('alpha_final', 0):.4e}\n")
            f.write(f"# Noise model: {self.pct_error*100:.1f}% + "
                    f"floor={self.noise_floor:.4f}\n")
            f.write("# x(m)\tz(m)\trho(Ohm.m)\tlog10_rho\tsensitivity\tresolution\n")
            if ib is not None and self._last_block_rho is not None:
                for bi, blk in enumerate(ib.blocks):
                    f.write(f"{blk['cx']:.2f}\t{blk['cz']:.2f}\t"
                            f"{self._last_block_rho[bi]:.3f}\t"
                            f"{np.log10(self._last_block_rho[bi]):.4f}\t"
                            f"{sens[bi]:.4f}\t{res_diag[bi]:.4f}\n")
            else:
                m = self.mesh
                for ic in range(min(m.n_cells, len(rho_result))):
                    ix = ic % m.ncx; iz = ic // m.ncx
                    si = sens[ic] if ic < len(sens) else 0
                    ri = res_diag[ic] if ic < len(res_diag) else 0
                    f.write(f"{m.x_cc[ix]:.2f}\t{m.z_cc[iz]:.2f}\t"
                            f"{rho_result[ic]:.3f}\t"
                            f"{np.log10(max(rho_result[ic], 0.1)):.4f}\t"
                            f"{si:.4f}\t{ri:.4f}\n")

        # ── 데이터 적합도 파일 ──
        with open(base + '_data.xyz', 'w') as f:
            f.write("# x(m)\tz(m)\tobs_rho_a\tcalc_rho_a\tmisfit(%)\tweighted_res\n")
            for i, ms in enumerate(self.survey.measurements):
                obs_i = d_obs[i]; calc_i = d_calc[i]
                misfit = (obs_i - calc_i) / max(abs(obs_i), 1e-10) * 100
                w_res = (np.log(abs(obs_i)) - np.log(max(abs(calc_i), 1e-10))) / \
                        self.data_errors[i] if hasattr(self, 'data_errors') else 0
                f.write(f"{ms['x']:.2f}\t{ms['z']:.2f}\t"
                        f"{obs_i:.4f}\t{calc_i:.4f}\t"
                        f"{misfit:.2f}\t{w_res:.3f}\n")

        # ── 수렴 이력 CSV ──
        conv = self.convergence
        with open(base + '_convergence.csv', 'w') as f:
            f.write("iteration,rms,chi2_N,alpha,phi_d,phi_m,roughness\n")
            for i in range(len(conv['rms'])):
                f.write(f"{i+1},{conv['rms'][i]:.6f},{conv['chi2'][i]:.6f},"
                        f"{conv['alpha'][i]:.6e},{conv['phi_d'][i]:.4f},"
                        f"{conv['phi_m'][i]:.4f},{conv['roughness'][i]:.4f}\n")

        return base


# ============================================================
# 시각화
# ============================================================

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 고비저항=빨강, 저비저항=파랑 — 중간톤(초록~노랑)을 넓게 배분
_RHO_COLORS = [
    (0.00, '#000080'),  # 진한 남색
    (0.06, '#0000CD'),  # 파랑
    (0.12, '#0066FF'),  # 밝은 파랑
    (0.18, '#00AAFF'),  # 하늘색
    (0.25, '#00DDCC'),  # 청록
    (0.32, '#00EE88'),  # 민트
    (0.40, '#44FF44'),  # 연두
    (0.48, '#99FF00'),  # 황록
    (0.55, '#CCFF00'),  # 연한 황록
    (0.62, '#FFFF00'),  # 노랑
    (0.70, '#FFDD00'),  # 골드
    (0.78, '#FFAA00'),  # 주황
    (0.85, '#FF6600'),  # 진한 주황
    (0.92, '#FF2200'),  # 빨강
    (1.00, '#990000'),  # 진한 빨강
]
RHO_CMAP = LinearSegmentedColormap.from_list(
    'resistivity', [(p, c) for p, c in _RHO_COLORS], N=512)


def run_star_inversion(survey, mesh, d_obs, rho_ref,
                       n_outer=3,
                       st_dip_weight=5.0,
                       st_smooth_sigma=3.0,
                       st_coherence_thresh=0.25,
                       use_mgs_init=True,
                       max_iter=8, tol=0.05,
                       noise_floor=0.001, pct_error=0.05,
                       target_chi2=1.0, cooling_factor=0.7,
                       callback=None,
                       init_dip_method='pseudosection_st',
                       nlevel_low_pct=30,
                       use_ood_gate=False,
                       ood_model_path=None,
                       ood_min_confidence=0.25,
                       ood_angle_blend=0.35):
    """Structure Tensor Adaptive Regularization (STAR) 역산.

    닭-달걀 딜레마를 외부 반복 루프(outer iteration)로 해결:
      1. 표준 L2 역산 → 초기 모델 (구조 텐서 초기화 불가 시)
      2. 수렴 모델에서 구조 텐서 추출 → 국소 경사각 필드
      3. 구조 텐서 WtW로 재역산 (이전 모델을 초기값으로 사용)
      4. 경사각 수렴까지 반복

    참고: Guenther et al. (2006), Linde et al. (2006)

    Parameters
    ----------
    survey   : DipDipSurvey
    mesh     : Mesh2D
    d_obs    : 관측 겉보기비저항 배열
    rho_ref  : 참조 비저항 (초기 모델)
    n_outer  : 외부 반복 횟수 (권장 3~5)
    st_dip_weight    : 경사 방향 평활화 강도 (>1, 권장 4~6)
    st_smooth_sigma  : 구조 텐서 가우시안 평활 σ (셀 단위, 권장 2~4)
    st_coherence_thresh : 구조 텐서 신뢰도 하한 (0~1, 낮을수록 더 많이 적용)
    use_ood_gate     : True이면 ML/OOD 진단 신뢰도와 국소 coherence로 이방성 강도를 게이트
    ood_model_path   : dip_ml_model.pkl 경로(None이면 RESIS_Pro.py와 현재 폴더에서 탐색)
    ood_min_confidence : 이 값 이하는 거의 등방 L2로 수축
    ood_angle_blend  : 견고한 전역 경사 추정값을 초기 theta field에 섞는 최대 비율

    Returns
    -------
    rho_final    : 최종 역산 비저항 (n_cells,)
    dip_history  : 외부 반복별 경사각 필드 리스트 [(ncz,ncx), ...]
    rms_history  : 외부 반복별 최종 RMS 리스트
    """
    from scipy.ndimage import gaussian_filter

    def _log(msg):
        if callback:
            callback(msg)

    star_gate_scalar = 1.0
    star_gate_info = None

    def _find_ood_model_path():
        if ood_model_path:
            return ood_model_path
        candidates = []
        try:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'dip_ml_model.pkl'))
        except Exception:
            pass
        candidates.append(os.path.join(os.getcwd(), 'dip_ml_model.pkl'))
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _apply_star_ood_gate(theta_mesh, conf_mesh, ps_median_dip):
        """전역 OOD confidence와 국소 coherence를 결합해 STAR 초기장/강도를 게이트."""
        if not use_ood_gate:
            return theta_mesh, conf_mesh, 1.0, None
        try:
            import pickle as _pickle
            from dip_diagnostics import diagnose_all as _diagnose_all
            from dip_ml_robust import robust_predict as _robust_predict

            model_path = _find_ood_model_path()
            if model_path is None:
                _log("  [OOD gate] dip_ml_model.pkl 없음 → 일반 STAR로 진행")
                return theta_mesh, conf_mesh, 1.0, None
            with open(model_path, 'rb') as _f:
                model_dict = _pickle.load(_f)

            diag = _diagnose_all(survey, d_obs, verbose=False)
            pred = _robust_predict(diag, model_dict, verbose=False)
            conf = float(pred.get('confidence', 0.0))
            gate = float(np.clip((conf - ood_min_confidence) /
                                 max(1.0 - ood_min_confidence, 1e-6), 0.0, 1.0))
            if float(pred.get('max_severity', 0.0)) > 2.0:
                gate *= 0.6

            # 신뢰도가 충분하면 ML/Fallback 전역 경사각을 초기 의사단면도 경사장에 약하게 혼합.
            # 공간 패턴은 theta_mesh가 유지하고, conf_mesh가 높은 영역에서만 보정한다.
            robust_dip = float(pred.get('estimate', 0.0))
            if robust_dip > 0.5 and np.isfinite(robust_dip):
                valid = np.abs(theta_mesh) > np.radians(0.5)
                sign = 1.0
                if np.any(valid):
                    sign = float(np.sign(np.mean(theta_mesh[valid])))
                    if sign == 0.0:
                        sign = 1.0
                blend = float(np.clip(ood_angle_blend * gate, 0.0, 0.75))
                conf_local = np.clip(conf_mesh, 0.0, 1.0) if conf_mesh is not None else np.ones_like(theta_mesh)
                theta_target = np.ones_like(theta_mesh) * np.radians(robust_dip) * sign
                theta_mesh = ((1.0 - blend * conf_local) * theta_mesh +
                              (blend * conf_local) * theta_target)

            _log("  [OOD gate] "
                 f"추정={pred.get('estimate', 0.0):.1f}° via {pred.get('method', 'NA')}, "
                 f"conf={conf:.2f}, OOD={pred.get('ood_score', 0.0):.2f}, "
                 f"gate={gate:.2f}, worst={pred.get('worst_feature', 'NA')}")
            return theta_mesh, conf_mesh, gate, pred
        except Exception as _e_gate:
            _log(f"  [OOD gate] 실패({_e_gate}) → 일반 STAR로 진행")
            return theta_mesh, conf_mesh, 1.0, None

    def _run_one(d_obs, rho_init_cells, WtW_override=None,
                 label='역산', run_label=''):
        """내부 역산 실행 헬퍼."""
        inv = Inversion2D(
            survey, mesh, rho_ref=rho_ref, alpha=1.0,
            max_iter=max_iter, tol=tol, solver_type='FDM',
            use_blocks=False, reg_type='L2',
            noise_floor=noise_floor, pct_error=pct_error,
            target_chi2=target_chi2, cooling_factor=cooling_factor,
            use_structure_tensor=False,
            st_dip_weight=st_dip_weight,
            st_smooth_sigma=st_smooth_sigma,
            initial_model=rho_init_cells,
            reference_model=rho_init_cells,
            skip_phase1_if_initial=True)   # WtW는 외부에서 주입

        # WtW를 직접 주입 (구조 텐서 내장 방식 대신 외부 루프 방식)
        if WtW_override is not None:
            if sparse.issparse(WtW_override):
                inv._WtW_aniso = WtW_override.toarray()
            else:
                inv._WtW_aniso = WtW_override

        rms_log = [999.0]
        def _cb(msg):
            s = str(msg)
            if 'RMS=' in s:
                try: rms_log[0] = float(s.split('RMS=')[1].split(',')[0])
                except: pass
            if callback:
                callback(s)

        rho_inv, hist, dcalc = inv.run(d_obs, callback=_cb, auto_alpha=True)
        return rho_inv, inv, rms_log[0]

    def _compute_st_WtW(inv_obj, m_cur):
        """수렴 모델에서 구조 텐서 WtW 계산."""
        # coherence_thresh를 잠시 덮어쓰기
        orig_thresh = 0.3
        theta_field = inv_obj._compute_structure_tensor_dips(m_cur)
        if theta_field is None:
            return None, None

        # coherence 재계산 (사용자 지정 thresh 적용)
        lnrho = m_cur.reshape(inv_obj.mesh.ncz, inv_obj.mesh.ncx)
        dlnrho_dx = np.gradient(lnrho, inv_obj.mesh.x_cc, axis=1)
        dlnrho_dz = np.gradient(lnrho, inv_obj.mesh.z_cc, axis=0)
        Jxx = gaussian_filter(dlnrho_dx**2, sigma=st_smooth_sigma)
        Jzz = gaussian_filter(dlnrho_dz**2, sigma=st_smooth_sigma)
        trace = Jxx + Jzz
        det  = Jxx * Jzz - gaussian_filter(dlnrho_dx*dlnrho_dz, sigma=st_smooth_sigma)**2
        disc = np.sqrt(np.maximum((trace/2)**2 - det, 0.0))
        lam1 = trace/2 + disc; lam2 = trace/2 - disc
        coherence = np.where(lam1 > 1e-10, (lam1-lam2)/(lam1+1e-10), 0.0)
        scale = np.clip((coherence - st_coherence_thresh) /
                        (1.0 - st_coherence_thresh), 0.0, 1.0)
        theta_field = theta_field * scale / np.clip(
            np.abs(inv_obj._st_dip_field) / (np.abs(theta_field) + 1e-10), 1e-10, 1.0) * scale

        # 원래 _st_dip_field 복원 후 coherence 적용
        theta_field = inv_obj._st_dip_field * scale

        WtW = inv_obj._build_structure_tensor_WtW(theta_field)
        return WtW, theta_field

    # ── Step 0: 의사단면도 기반 구조 텐서 초기화 ──
    # bootstrap 문제 해결: 역산 전 데이터 자체에서 지층 경사 초기 추정
    # 의사단면도(x_mid, z_pseudo, rho_a)를 규칙 격자에 보간 → 구조 텐서
    _log("  [STAR 초기화] 의사단면도에서 구조 텐서 초기 추정...")
    WtW_pseudo = None
    dip_pseudo = None
    conf_pseudo = None
    try:
        from scipy.interpolate import griddata as _griddata

        # 의사단면도 좌표
        xs = np.array([m['x'] for m in survey.measurements])
        zs = np.array([m['z'] for m in survey.measurements])
        log_ra = np.log10(np.maximum(d_obs, 1.0))

        # 규칙 격자 보간 (전극 범위)
        x1_ps = survey.electrode_x[0]; x2_ps = survey.electrode_x[-1]
        z_max_ps = survey.n_max * survey.a
        xi_ps = np.linspace(x1_ps, x2_ps, 80)
        zi_ps = np.linspace(0, z_max_ps, 40)
        XI_ps, ZI_ps = np.meshgrid(xi_ps, zi_ps)
        pseudo_grid = _griddata((xs, zs), log_ra,
                                (XI_ps, ZI_ps), method='linear')
        # 보간 불가 영역 nearest로 채움
        nan_mask = np.isnan(pseudo_grid)
        if nan_mask.any():
            pseudo_grid[nan_mask] = _griddata(
                (xs, zs), log_ra,
                (XI_ps[nan_mask], ZI_ps[nan_mask]), method='nearest')

        # 의사단면도 구조 텐서
        dx_ps = np.gradient(pseudo_grid, xi_ps, axis=1)
        dz_ps = np.gradient(pseudo_grid, zi_ps, axis=0)
        sig_ps = max(st_smooth_sigma, 2.0)
        Jxx_ps = gaussian_filter(dx_ps**2, sigma=sig_ps)
        Jzz_ps = gaussian_filter(dz_ps**2, sigma=sig_ps)
        Jxz_ps = gaussian_filter(dx_ps*dz_ps, sigma=sig_ps)

        trace_ps = Jxx_ps + Jzz_ps
        det_ps = Jxx_ps*Jzz_ps - Jxz_ps**2
        disc_ps = np.sqrt(np.maximum((trace_ps/2)**2 - det_ps, 0.0))
        lam1_ps = trace_ps/2 + disc_ps
        lam2_ps = trace_ps/2 - disc_ps

        vx_ps = Jxz_ps; vz_ps = lam1_ps - Jxx_ps
        norm_ps = np.sqrt(vx_ps**2 + vz_ps**2) + 1e-10
        flip_ps = vz_ps < 0
        vx_c = np.where(flip_ps, -vx_ps, vx_ps)
        vz_c = np.where(flip_ps, -vz_ps, vz_ps)
        theta_ps_grid = np.arctan2(-vx_c/norm_ps, vz_c/norm_ps)

        # 신뢰도 필터
        coh_ps = np.where(lam1_ps > 1e-10, (lam1_ps-lam2_ps)/(lam1_ps+1e-10), 0.0)
        scale_ps = np.clip((coh_ps - st_coherence_thresh)/(1.0 - st_coherence_thresh), 0.0, 1.0)
        theta_ps_grid *= scale_ps

        # 의사단면도 격자 → FDM 격자로 보간
        xi_flat = XI_ps.ravel(); zi_flat = ZI_ps.ravel()
        theta_flat = theta_ps_grid.ravel()
        conf_flat = scale_ps.ravel()
        XX, ZZ = np.meshgrid(mesh.x_cc, mesh.z_cc)
        theta_mesh = _griddata((xi_flat, zi_flat), theta_flat,
                               (XX, ZZ), method='linear', fill_value=0.0)
        conf_mesh = _griddata((xi_flat, zi_flat), conf_flat,
                              (XX, ZZ), method='linear', fill_value=0.0)
        nan_m = np.isnan(theta_mesh)
        if nan_m.any():
            theta_mesh[nan_m] = 0.0
        nan_c = np.isnan(conf_mesh)
        if nan_c.any():
            conf_mesh[nan_c] = 0.0
        conf_mesh = np.clip(conf_mesh, 0.0, 1.0)

        # 유의미한 추정 통계
        x1 = survey.electrode_x[0]; x2 = survey.electrode_x[-1]
        ix0 = np.searchsorted(mesh.x_cc, x1)
        ix1_c = np.searchsorted(mesh.x_cc, x2)
        iz1 = np.searchsorted(mesh.z_cc, survey.n_max * survey.a * 1.5)
        core = np.degrees(theta_mesh[:iz1, ix0:ix1_c])
        nonzero = core[np.abs(core) > 0.5]
        ps_median_dip = 0.0
        if len(nonzero) > 0:
            ps_median_dip = float(np.median(np.abs(nonzero)))
            _log(f"  의사단면도 경사 추정: 중앙값={ps_median_dip:.1f}°, "
                 f"75%ile={np.percentile(np.abs(nonzero), 75):.1f}°")

        # ── n-레벨 중심추적 (Option A/B 지원) ──
        nlevel_dip = None
        nlevel_R2 = 0.0
        try:
            cents = []
            for n_i in range(1, survey.n_max + 1):
                msk = np.array([m['n'] == n_i for m in survey.measurements])
                if msk.sum() < 4: continue
                x_pts = np.array([m['x'] for m in survey.measurements])[msk]
                rho_pts = np.asarray(d_obs)[msk]
                thr = np.percentile(rho_pts, nlevel_low_pct)
                sel = rho_pts <= thr
                if sel.sum() < 3: continue
                x_sel = x_pts[sel]; rho_sel = rho_pts[sel]
                w_ = 1.0 / (rho_sel + 1e-6)
                x_c = np.sum(x_sel * w_) / np.sum(w_)
                z_p = n_i * survey.a * 0.519
                cents.append((n_i, x_c, z_p))
            if len(cents) >= 3:
                xs_ = np.array([c[1] for c in cents])
                zs_ = np.array([c[2] for c in cents])
                slope_, icpt_ = np.polyfit(xs_, zs_, 1)
                nlevel_dip = float(np.degrees(np.arctan(np.abs(slope_))))
                # R²
                x_pred = (zs_ - icpt_)/slope_ if abs(slope_) > 1e-6 else xs_
                ss_res = np.sum((xs_ - x_pred)**2)
                ss_tot = np.sum((xs_ - np.mean(xs_))**2)
                nlevel_R2 = max(0.0, 1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
                _log(f"  n-레벨 중심추적: {nlevel_dip:.1f}° (R²={nlevel_R2:.2f})")
        except Exception as _e2:
            _log(f"  n-레벨 추적 실패: {_e2}")

        # ── 방법별 theta_mesh 조정 ──
        if init_dip_method == 'nlevel_uniform' and nlevel_dip is not None:
            # Option A: theta_mesh를 균일값으로 대체 (부호는 의사단면도 ST에서 유추)
            valid = np.abs(theta_mesh) > 0.005
            sign_ = 1.0
            if valid.any():
                sign_ = float(np.sign(np.mean(theta_mesh[valid])))
                if sign_ == 0: sign_ = 1.0
            theta_mesh = np.ones_like(theta_mesh) * np.radians(nlevel_dip) * sign_
            _log(f"  [Option A] 균일 경사장 대체: {nlevel_dip:.1f}° (부호={int(sign_):+d})")
        elif init_dip_method == 'ensemble' and nlevel_dip is not None and ps_median_dip > 0.5:
            # Option B: 의사단면도 공간 패턴 유지 + 중앙값을 앙상블로 스케일
            # 가중치: coherence 1.0 (의사단면도 ST), R² (n-레벨)
            ens_dip = (ps_median_dip + nlevel_R2 * nlevel_dip) / (1.0 + nlevel_R2)
            scale_factor = ens_dip / ps_median_dip
            theta_mesh = theta_mesh * scale_factor
            _log(f"  [Option B] 앙상블 스케일링: {ps_median_dip:.1f}° → {ens_dip:.1f}° "
                 f"(×{scale_factor:.2f}, R²가중={nlevel_R2:.2f})")

        theta_mesh, conf_mesh, star_gate_scalar, star_gate_info = _apply_star_ood_gate(
            theta_mesh, conf_mesh, ps_median_dip)

        # 임시 Inversion2D로 WtW 구성 (st_dip_weight 사용)
        _dummy_inv = Inversion2D(
            survey, mesh, rho_ref=rho_ref, use_blocks=False,
            st_dip_weight=st_dip_weight, st_smooth_sigma=st_smooth_sigma)
        WtW_pseudo = _dummy_inv._build_structure_tensor_WtW(
            theta_mesh,
            confidence_field=conf_mesh if use_ood_gate else None,
            confidence_scalar=star_gate_scalar)
        dip_pseudo = theta_mesh
        conf_pseudo = conf_mesh
        _log("  의사단면도 구조 텐서 초기화 완료.")
    except Exception as _e:
        _log(f"  의사단면도 초기화 실패 ({_e}), 표준 L2로 시작.")
        WtW_pseudo = None

    # ── Step 1: MGS 초기 역산으로 선명한 경계 추출 (선택) ──
    # L2보다 MGS가 경계를 더 명확히 복원 → 구조 텐서 추정 정확도 향상
    mgs_dip_field = None
    if use_mgs_init:
        _log("  [MGS 초기화] 경계 강조 역산으로 구조 텐서 초기 추정...")
        try:
            mgs_inv = Inversion2D(
                survey, mesh, rho_ref=rho_ref, alpha=1.0,
                max_iter=max_iter, tol=tol, solver_type='FDM',
                use_blocks=False, reg_type='MGS',
                noise_floor=noise_floor, pct_error=pct_error,
                target_chi2=target_chi2, cooling_factor=cooling_factor,
                use_structure_tensor=False)
            rms_mgs = [999.0]
            def _mgs_cb(msg):
                s = str(msg)
                if 'RMS=' in s:
                    try: rms_mgs[0] = float(s.split('RMS=')[1].split(',')[0])
                    except: pass
            rho_mgs, _, _ = mgs_inv.run(d_obs, callback=_mgs_cb, auto_alpha=True)
            m_mgs = np.log(np.clip(rho_mgs, 0.1, 1e5))
            mgs_dip_field = mgs_inv._compute_structure_tensor_dips(m_mgs)
            mgs_conf_field = getattr(mgs_inv, '_st_confidence_field', None)

            if mgs_dip_field is not None:
                # MGS + 의사단면도 텐서 결합: 신뢰도 기반 가중 평균
                # MGS 모델이 경계가 선명 → 경계 근처에서 신뢰도 높음
                # 의사단면도 텐서는 전체 구조 방향 참고용
                lnrho_mgs = m_mgs.reshape(mesh.ncz, mesh.ncx)
                grad_mag = np.sqrt(
                    np.gradient(lnrho_mgs, mesh.x_cc, axis=1)**2 +
                    np.gradient(lnrho_mgs, mesh.z_cc, axis=0)**2)
                # 기울기 크기 정규화 (0~1)
                gmax = np.percentile(grad_mag, 95) + 1e-10
                grad_weight = np.clip(grad_mag / gmax, 0.0, 1.0)

                # 의사단면도 텐서와 MGS 텐서 결합
                if dip_pseudo is not None:
                    theta_combined = (grad_weight * mgs_dip_field +
                                      (1.0 - grad_weight) * dip_pseudo)
                    if conf_pseudo is not None and mgs_conf_field is not None:
                        conf_combined = (grad_weight * mgs_conf_field +
                                         (1.0 - grad_weight) * conf_pseudo)
                    elif mgs_conf_field is not None:
                        conf_combined = mgs_conf_field
                    else:
                        conf_combined = conf_pseudo
                else:
                    theta_combined = mgs_dip_field
                    conf_combined = mgs_conf_field

                # 통계
                x1 = survey.electrode_x[0]; x2 = survey.electrode_x[-1]
                ix0_s = np.searchsorted(mesh.x_cc, x1)
                ix1_s = np.searchsorted(mesh.x_cc, x2)
                iz1_s = np.searchsorted(mesh.z_cc, survey.n_max * survey.a * 1.5)
                core_mgs = np.degrees(theta_combined[:iz1_s, ix0_s:ix1_s])
                sig_mgs = core_mgs[np.abs(core_mgs) > 0.5]
                if len(sig_mgs) > 0:
                    _log(f"  MGS+의사단면도 경사 추정: "
                         f"중앙값={np.median(np.abs(sig_mgs)):.1f}°, "
                         f"75%ile={np.percentile(np.abs(sig_mgs), 75):.1f}°")

                # 결합된 구조 텐서 WtW 구성
                _tmp_inv = Inversion2D(
                    survey, mesh, rho_ref=rho_ref, use_blocks=False,
                    st_dip_weight=st_dip_weight)
                WtW_pseudo = _tmp_inv._build_structure_tensor_WtW(
                    theta_combined,
                    confidence_field=conf_combined if use_ood_gate else None,
                    confidence_scalar=star_gate_scalar)
                dip_pseudo = theta_combined
                conf_pseudo = conf_combined
                _log(f"  MGS 초기화 완료 (RMS={rms_mgs[0]:.4f})")
        except Exception as _e_mgs:
            _log(f"  MGS 초기화 실패 ({_e_mgs}), 의사단면도 텐서만 사용.")

    # ── 외부 반복 루프 ──
    dip_history = []
    rms_history = []
    WtW_cur = WtW_pseudo   # 의사단면도+MGS 텐서로 시작
    rho_prev = None

    for outer in range(n_outer):
        _log(f"━━ STAR 외부 반복 {outer+1}/{n_outer} ━━")

        if outer == 0 and WtW_cur is not None:
            _log("  [MGS+의사단면도 텐서] 데이터 기반 이방성 정규화...")
        elif outer == 0:
            _log("  [초기 L2] 표준 역산으로 초기 구조 추출...")
        else:
            _log(f"  [구조텐서] 이전 모델 기반 이방성 정규화 적용...")

        rho_inv, inv_obj, rms = _run_one(
            d_obs, rho_prev, WtW_override=WtW_cur,
            label=f'STAR outer-{outer+1}')

        rms_history.append(rms)
        _log(f"  외부 반복 {outer+1} 완료: RMS={rms:.4f}")

        # 수렴 모델에서 구조 텐서 추출
        m_conv = np.log(np.clip(rho_inv, 0.1, 1e5))

        # Inversion2D 인스턴스에 구조 텐서 메서드 활용
        theta_field = inv_obj._compute_structure_tensor_dips(m_conv)

        if theta_field is not None:
            dip_history.append(theta_field.copy())

            # 추정 경사각 통계 (코어 영역)
            x1, x2 = survey.electrode_x[0], survey.electrode_x[-1]
            ix0 = np.searchsorted(mesh.x_cc, x1)
            ix1 = np.searchsorted(mesh.x_cc, x2)
            iz1 = np.searchsorted(mesh.z_cc, survey.n_max * survey.a * 1.5)
            core_dip = np.degrees(theta_field[:iz1, ix0:ix1])
            # 유의미한 셀만 (coherence > 0): scale 0이 아닌 셀
            nonzero = core_dip[np.abs(core_dip) > 0.5]
            if len(nonzero) > 0:
                med_dip = float(np.median(np.abs(nonzero)))
                p75_dip = float(np.percentile(np.abs(nonzero), 75))
                _log(f"  코어 경사각 추정: 중앙값={med_dip:.1f}°, 75%ile={p75_dip:.1f}°")
            else:
                _log("  코어 경사각: 유의미한 신호 없음 (등방 구조 추정)")

            # 다음 외부 반복을 위한 WtW 갱신
            # (coherence 적용은 _compute_structure_tensor_dips 내부에서 처리)
            WtW_cur = inv_obj._build_structure_tensor_WtW(
                theta_field,
                confidence_field=getattr(inv_obj, '_st_confidence_field', None) if use_ood_gate else None,
                confidence_scalar=star_gate_scalar)

            # 외부 반복 수렴 판정: 경사각 변화량
            if len(dip_history) >= 2:
                dip_change = float(np.mean(np.abs(
                    dip_history[-1] - dip_history[-2])))
                dip_change_deg = np.degrees(dip_change)
                _log(f"  경사각 변화: {dip_change_deg:.2f}°")
                if dip_change_deg < 1.0:
                    _log("  ✓ 경사각 수렴 → 외부 루프 종료")
                    rho_prev = rho_inv
                    break
        else:
            dip_history.append(None)

        rho_prev = rho_inv

    _log(f"━━ STAR 완료 (외부 반복 {len(rms_history)}회) ━━")
    return rho_inv, dip_history, rms_history


def _trapezoid_mask(XI, ZI, survey):
    """데이터 커버리지 영역의 사다리꼴 마스크 생성"""
    a = survey.a
    ex = survey.electrode_x
    x0, x1 = ex[0], ex[-1]
    # 각 n-level에서의 x 범위 계산
    n_levels = sorted(set(m['n'] for m in survey.measurements))
    x_mins, x_maxs, zs = [], [], []
    for n in n_levels:
        pts = [(m['x'], m['z']) for m in survey.measurements if m['n'] == n]
        xs = [p[0] for p in pts]
        x_mins.append(min(xs)); x_maxs.append(max(xs)); zs.append(n * a)

    # 사다리꼴 경계 (좌상 → 좌하 → 우하 → 우상)
    verts = ([(x_mins[0] - a * 0.3, zs[0] - a * 0.5)] +
             [(xm - a * 0.3, z) for xm, z in zip(x_mins, zs)] +
             [(x_mins[-1] - a * 0.3, zs[-1] + a * 0.5)] +
             [(x_maxs[-1] + a * 0.3, zs[-1] + a * 0.5)] +
             [(xm + a * 0.3, z) for xm, z in zip(x_maxs[::-1], zs[::-1])] +
             [(x_maxs[0] + a * 0.3, zs[0] - a * 0.5)])
    path = Path(verts)
    pts = np.column_stack([XI.ravel(), ZI.ravel()])
    return path.contains_points(pts).reshape(XI.shape), path


def plot_pseudosection(ax, survey, rho_a, title="의사단면도",
                       cmap=None, show_values=True, global_vrange=None):
    if cmap is None:
        cmap = RHO_CMAP
    a = survey.a
    x = np.array([m['x'] for m in survey.measurements])
    z = np.array([m['z'] for m in survey.measurements])
    vals = np.maximum(np.abs(rho_a), 0.1)

    # 고해상도 보간 격자
    xi = np.linspace(x.min() - a, x.max() + a, 400)
    zi = np.linspace(z.min() - a * 0.5, z.max() + a * 0.5, 200)
    XI, ZI = np.meshgrid(xi, zi)

    log_vals = np.log10(vals)
    VI_log = griddata((x, z), log_vals, (XI, ZI), method='cubic')
    VI_fill = griddata((x, z), log_vals, (XI, ZI), method='linear')
    nan_mask = np.isnan(VI_log)
    VI_log[nan_mask] = VI_fill[nan_mask]

    # 사다리꼴 마스크 적용
    trap_mask, trap_path = _trapezoid_mask(XI, ZI, survey)
    VI_log[~trap_mask] = np.nan
    VI_lin = 10 ** VI_log

    # 색상 레인지: 전체 범위 사용 (global_vrange 지정 시 통일)
    if global_vrange:
        vmin, vmax = global_vrange
    else:
        vmin, vmax = vals.min(), vals.max()
    norm = LogNorm(vmin=max(vmin, 0.1), vmax=vmax)
    im = ax.pcolormesh(XI, ZI, VI_lin, cmap=cmap, norm=norm,
                       shading='gouraud', rasterized=True)

    # 측정점 + 비저항 값 표시
    ax.scatter(x, z, c=vals, cmap=cmap, norm=norm,
               edgecolors='k', s=14, linewidths=0.3, zorder=5)
    if show_values:
        for xi_, zi_, vi_ in zip(x, z, vals):
            ax.text(xi_, zi_, f'{vi_:.0f}', fontsize=4.5, ha='center', va='center',
                    fontweight='bold', zorder=6)

    # 전극 번호 표시 (겹침 방지: 간격에 따라 표시 간격 조정)
    ex = survey.electrode_x
    n_elec = len(ex)
    label_step = 1 if n_elec <= 15 else (2 if n_elec <= 30 else 5)
    for i, xe in enumerate(ex):
        ax.plot(xe, -a * 0.05, 'kv', markersize=3, zorder=10, clip_on=False)
        if (i % label_step == 0) or (i == n_elec - 1):
            ax.text(xe, -a * 0.65, str(i + 1), fontsize=5, ha='center', va='bottom',
                    fontweight='bold', clip_on=False)

    ax.set_xlim(ex[0] - a, ex[-1] + a)
    ax.set_ylim(z.max() + a * 0.8, -a * 1.2)
    ax.set_xlabel('거리 (m)'); ax.set_ylabel('의사깊이 (m)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')

    # 수평 컬러바
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.55)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('비저항 (Ω·m)')
    return im


def plot_misfit_pseudosection(ax, survey, d_obs, d_calc, title="잔차 (%)"):
    """관측값 대비 잔차를 의사단면도로 표시.
    misfit(%) = (d_obs - d_calc) / d_obs * 100
    """
    a = survey.a
    x = np.array([m['x'] for m in survey.measurements])
    z = np.array([m['z'] for m in survey.measurements])
    d_obs_abs = np.maximum(np.abs(d_obs), 1e-10)
    misfit_pct = (d_obs - d_calc) / d_obs_abs * 100.0

    # 보간 격자
    xi = np.linspace(x.min() - a, x.max() + a, 400)
    zi = np.linspace(z.min() - a * 0.5, z.max() + a * 0.5, 200)
    XI, ZI = np.meshgrid(xi, zi)

    VI = griddata((x, z), misfit_pct, (XI, ZI), method='cubic')
    VI_fill = griddata((x, z), misfit_pct, (XI, ZI), method='linear')
    nan_mask = np.isnan(VI)
    VI[nan_mask] = VI_fill[nan_mask]

    trap_mask, trap_path = _trapezoid_mask(XI, ZI, survey)
    VI[~trap_mask] = np.nan

    # 대칭 컬러맵: 파랑(음)=계산이 큼, 빨강(양)=관측이 큼
    vabs = max(np.nanpercentile(np.abs(misfit_pct), 95), 5.0)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.pcolormesh(XI, ZI, VI, cmap='RdBu_r', norm=norm,
                       shading='gouraud', rasterized=True)

    # 측정점
    ax.scatter(x, z, c=misfit_pct, cmap='RdBu_r', norm=norm,
               edgecolors='k', s=14, linewidths=0.3, zorder=5)

    # 전극 표시
    ex = survey.electrode_x; n_elec = len(ex)
    label_step = 1 if n_elec <= 15 else (2 if n_elec <= 30 else 5)
    for i, xe in enumerate(ex):
        ax.plot(xe, -a * 0.05, 'kv', markersize=3, zorder=10, clip_on=False)
        if (i % label_step == 0) or (i == n_elec - 1):
            ax.text(xe, -a * 0.65, str(i + 1), fontsize=5, ha='center', va='bottom',
                    fontweight='bold', clip_on=False)

    ax.set_xlim(ex[0] - a, ex[-1] + a)
    ax.set_ylim(z.max() + a * 0.8, -a * 1.2)
    ax.set_xlabel('거리 (m)'); ax.set_ylabel('의사깊이 (m)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.55)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('잔차 (%)')

    # 통계 표시
    rms = np.sqrt(np.mean(misfit_pct**2))
    mean_abs = np.mean(np.abs(misfit_pct))
    ax.text(0.02, 0.98, f'RMS: {rms:.1f}%  |  평균|잔차|: {mean_abs:.1f}%',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return im


def plot_obs_vs_calc(ax, d_obs, d_calc, title="관측값 vs 계산값"):
    """산점도: 관측 겉보기비저항 vs 계산 겉보기비저항"""
    ax.scatter(np.abs(d_obs), np.abs(d_calc), s=15, alpha=0.6,
               edgecolors='k', linewidths=0.3, c='steelblue')
    # 1:1 라인
    all_vals = np.concatenate([np.abs(d_obs), np.abs(d_calc)])
    vmin, vmax = all_vals.min() * 0.5, all_vals.max() * 2
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='1:1')
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('관측 겉보기비저항 (Ω·m)')
    ax.set_ylabel('계산 겉보기비저항 (Ω·m)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # R² 계산
    log_obs = np.log10(np.maximum(np.abs(d_obs), 1e-10))
    log_calc = np.log10(np.maximum(np.abs(d_calc), 1e-10))
    ss_res = np.sum((log_obs - log_calc)**2)
    ss_tot = np.sum((log_obs - np.mean(log_obs))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    rms_pct = np.sqrt(np.mean(((d_obs - d_calc) / np.maximum(np.abs(d_obs), 1e-10))**2)) * 100
    ax.text(0.05, 0.92, f'R² = {r2:.4f}\nRMS = {rms_pct:.1f}%',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def plot_model_section(ax, mesh, rho, survey, title="비저항 단면도",
                       cmap=None, show_values=False, global_vrange=None,
                       show_block_grid=False, inv_blocks=None,
                       block_rho=None, clip_electrodes=5,
                       contour_levels=None):
    """contour_levels: 등치선 비저항 값 리스트 (예: [50, 100, 200, 500])"""
    if cmap is None:
        cmap = RHO_CMAP
    a = survey.a
    ex = survey.electrode_x
    has_topo = mesh.has_topo
    z_max_model = survey.n_max * a * 1.25

    # 보간 소스: 블록 중심 (부드러움) 또는 셀 중심
    if block_rho is not None and inv_blocks is not None:
        src_x = np.array([b['cx'] for b in inv_blocks.blocks])
        src_z = np.array([b['cz'] for b in inv_blocks.blocks])
        src_rho = block_rho
    else:
        ncx, ncz = mesh.ncx, mesh.ncz
        rho_2d = rho.reshape(ncz, ncx)
        x_lim = (ex[0] - a, ex[-1] + a)
        z_lim = survey.n_max * a * 1.3
        mask_x = (mesh.x_nodes >= x_lim[0]) & (mesh.x_nodes <= x_lim[1])
        mask_z = mesh.z_nodes <= z_lim
        ix0 = np.where(mask_x)[0][0]; ix1 = np.where(mask_x)[0][-1]
        iz1 = np.where(mask_z)[0][-1]
        cx = mesh.x_cc[ix0:ix1]; cz = mesh.z_cc[:iz1]
        rho_sub = rho_2d[:iz1, ix0:ix1]
        # 셀 모드 시각화: 로그 공간 Gaussian + 서브샘플링으로 부드러운 표시
        # (MGS 결과의 블록형 경계를 시각적으로 부드럽게, 역산값 변경 없음)
        from scipy.ndimage import gaussian_filter
        log_sub = np.log10(np.maximum(rho_sub, 0.1))
        log_sub = gaussian_filter(log_sub, sigma=2.5)
        rho_sub_s = 10 ** log_sub
        # 셀 간격 ≈ a/4 → 블록 수준 소스 밀도로 서브샘플 (4셀 ≈ 1 전극간격)
        sx = max(1, round(a / (cx[1] - cx[0]) / 2)) if len(cx) > 1 else 1
        sz = max(1, round(a / (cz[1] - cz[0]) / 2)) if len(cz) > 1 else 1
        cx_s = cx[::sx]; cz_s = cz[::sz]
        rho_ss = rho_sub_s[::sz, ::sx]
        CX, CZ = np.meshgrid(cx_s, cz_s)
        src_x = CX.ravel(); src_z = CZ.ravel()
        src_rho = rho_ss.ravel()

    # 상/하부 경계 포인트 추가 (격자 패턴 방지)
    z_max_model = survey.n_max * a * 1.25
    extra_x, extra_z, extra_rho = [], [], []

    # 최상부: z=0 포인트 추가
    top_mask = src_z < (src_z.min() + a * 0.6)
    if top_mask.sum() > 0:
        extra_x.append(src_x[top_mask])
        extra_z.append(np.zeros(top_mask.sum()))
        extra_rho.append(src_rho[top_mask])

    # 최하부: z_max 포인트 추가
    bot_mask = src_z > (src_z.max() - a * 0.6)
    if bot_mask.sum() > 0:
        extra_x.append(src_x[bot_mask])
        extra_z.append(np.full(bot_mask.sum(), z_max_model))
        extra_rho.append(src_rho[bot_mask])

    if extra_x:
        src_x = np.concatenate([src_x] + extra_x)
        src_z = np.concatenate([src_z] + extra_z)
        src_rho = np.concatenate([src_rho] + extra_rho)

    if global_vrange:
        vmin, vmax = global_vrange
    else:
        vmin, vmax = src_rho.min(), src_rho.max()
    norm = LogNorm(vmin=max(vmin, 0.1), vmax=vmax)

    if has_topo:
        from scipy.interpolate import interp1d
        elev_func = interp1d(mesh.x_nodes, mesh.surface_elev,
                             kind='linear', fill_value='extrapolate')
        ylabel = '표고 (m)'; elec_elev = elev_func(ex)
        elev_max = mesh.max_elev + a * 0.5
    else:
        elev_func = None; ylabel = '깊이 (m)'
        elec_elev = np.zeros_like(ex)

    # ── 표시 영역: clip_electrodes개 잘라낸 직사각형 또는 전체 ──
    log_src = np.log10(np.maximum(src_rho, 0.1))
    n_cut = max(0, int(clip_electrodes))
    if n_cut > 0 and n_cut < len(ex) // 2:
        x_left = ex[n_cut]
        x_right = ex[-(n_cut + 1)]
    else:
        x_left = ex[0] - a * 0.3
        x_right = ex[-1] + a * 0.3

    # 마스크: 잘림 위치(x_left~x_right)는 직사각형,
    # 바깥(전극 1~n_cut, 끝~n_cut)은 깊이에 따라 경사 확장
    slope = 0.7  # 경사율
    x_outer_left = ex[0]    # 전체 측선 시작
    x_outer_right = ex[-1]  # 전체 측선 끝

    if has_topo:
        src_elev = elev_func(src_x) - src_z
        xi = np.linspace(x_outer_left - a, x_outer_right + a, 500)
        surf_at_xi = elev_func(xi)
        elev_min = min(elev_func(x_left), elev_func(x_right)) - z_max_model - a
        zi_elev = np.linspace(elev_min, elev_max, 300)
        XI, ZI_elev = np.meshgrid(xi, zi_elev)
        VI = griddata((src_x, src_elev), log_src, (XI, ZI_elev), method='cubic')
        VI_fill = griddata((src_x, src_elev), log_src, (XI, ZI_elev), method='nearest')
        VI[np.isnan(VI)] = VI_fill[np.isnan(VI)]
        for j in range(len(xi)):
            VI[ZI_elev[:, j] > surf_at_xi[j], j] = np.nan
            VI[ZI_elev[:, j] < (surf_at_xi[j] - z_max_model), j] = np.nan
        # 좌측: x_left에서 직선, 바깥쪽은 깊이에 따라 경사
        depth_from_top = mesh.max_elev - ZI_elev
        left_bound = np.where(XI <= x_left, x_left - depth_from_top * slope, x_left)
        right_bound = np.where(XI >= x_right, x_right + depth_from_top * slope, x_right + 999)
        VI[(XI < left_bound) | (XI > right_bound)] = np.nan
        im = ax.pcolormesh(XI, ZI_elev, 10**VI, cmap=cmap, norm=norm,
                           shading='gouraud', rasterized=True)
        ax.plot(xi, surf_at_xi, 'k-', lw=1.5, zorder=8)
        ax.fill_between(xi, surf_at_xi, elev_max + a, color='white', zorder=7)
        ylim_bot = elev_min + a * 0.5; ylim_top = elev_max
    else:
        xi = np.linspace(x_outer_left - a, x_outer_right + a, 500)
        zi = np.linspace(0, z_max_model, 300)
        XI, ZI_elev = np.meshgrid(xi, zi)
        VI = griddata((src_x, src_z), log_src, (XI, ZI_elev), method='cubic')
        VI_fill = griddata((src_x, src_z), log_src, (XI, ZI_elev), method='nearest')
        VI[np.isnan(VI)] = VI_fill[np.isnan(VI)]
        if n_cut >= 4:
            # 5개 이상: 직사각형
            mask_out = (XI < x_left) | (XI > x_right)
        else:
            # 4개 이하: 전체 범위에서 경사
            slope = 0.7
            left_bound = x_outer_left + ZI_elev * slope
            right_bound = x_outer_right - ZI_elev * slope
            mask_out = (XI < left_bound) | (XI > right_bound)
        VI[mask_out] = np.nan
        im = ax.pcolormesh(XI, ZI_elev, 10**VI, cmap=cmap, norm=norm,
                           shading='gouraud', rasterized=True)
        ylim_bot = z_max_model + a * 0.3; ylim_top = -a * 0.6

    # ── 등치선 (contour) 오버레이 ──
    if contour_levels is not None and len(contour_levels) > 0:
        try:
            cs = ax.contour(XI, ZI_elev, 10**VI, levels=sorted(contour_levels),
                            colors='black', linewidths=0.8, zorder=12)
            ax.clabel(cs, fmt='%g', fontsize=6, inline=True)
        except Exception:
            pass  # 보간 실패 시 무시

    # ── 블록 격자선 오버레이 (선택) ──
    if show_block_grid and inv_blocks is not None:
        inv_blocks.plot_blocks(ax, show_grid_only=True)

    # ── 비저항 값 표시 (블록 중심에 숫자) ──
    if show_values and block_rho is not None and inv_blocks is not None:
        for bi, blk in enumerate(inv_blocks.blocks):
            bx, bz = blk['cx'], blk['cz']
            if bx < x_left or bx > x_right or bz > z_max_model:
                continue
            rv = block_rho[bi]
            ax.text(bx, bz, f'{rv:.0f}', fontsize=7, ha='center', va='center',
                    color='black', fontweight='bold', zorder=15)
    elif show_values and block_rho is None:
        # 셀 모드: 적당한 간격으로 값 표시
        ncx, ncz = mesh.ncx, mesh.ncz
        rho_2d = rho.reshape(ncz, ncx)
        step_x = max(1, ncx // 20); step_z = max(1, ncz // 10)
        for iz in range(0, ncz, step_z):
            for ix in range(0, ncx, step_x):
                xp, zp = mesh.x_cc[ix], mesh.z_cc[iz]
                if xp < x_left or xp > x_right or zp > z_max_model:
                    continue
                rv = rho_2d[iz, ix]
                ax.text(xp, zp, f'{rv:.0f}', fontsize=6, ha='center', va='center',
                        color='black', fontweight='bold', zorder=15)

    # 전극 표시 (단면도 위에 배치, 겹침 방지)
    n_elec = len(ex)
    label_step = 1 if n_elec <= 15 else (2 if n_elec <= 30 else 5)
    elec_y = -a * 0.05 if not has_topo else 0  # 마커 위치 (표면 바로 위)
    label_y = -a * 0.35 if not has_topo else a * 0.3  # 번호: 마커 바로 위
    for i, xe in enumerate(ex):
        ey = elec_elev[i] if has_topo else elec_y
        ly = (elec_elev[i] + label_y) if has_topo else label_y
        ax.plot(xe, ey, 'kv', markersize=3, zorder=10, clip_on=False)
        if (i % label_step == 0) or (i == n_elec - 1):
            ax.text(xe, ly, str(i + 1), fontsize=5, ha='center',
                    va='bottom', fontweight='bold', clip_on=False)

    ax.set_xlim(x_outer_left - a * 0.5, x_outer_right + a * 0.5)
    ax.set_ylim(ylim_bot, -a * 0.7 if not has_topo else ylim_top)
    ax.set_xlabel('거리 (m)'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')

    # 수평 컬러바
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.45)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('비저항 (Ω·m)')

    # 마우스 이동 시 비저항 값 표시
    _xi = xi; _zi = zi if not has_topo else zi_elev; _VI = VI
    def _fmt_coord(x, y):
        # 보간 격자에서 가장 가까운 인덱스 찾기
        try:
            ix = np.argmin(np.abs(_xi - x))
            iz = np.argmin(np.abs(_zi - y))
            val = _VI[iz, ix]
            if np.isnan(val):
                return f'x={x:.1f}m, {ylabel}={y:.1f}m'
            rho_val = 10 ** val
            return f'x={x:.1f}m, {ylabel}={y:.1f}m, ρ={rho_val:.0f} Ω·m'
        except:
            return f'x={x:.1f}m, {ylabel}={y:.1f}m'
    ax.format_coord = _fmt_coord

    return im


# ============================================================
# 학술 분석 시각화 함수
# ============================================================

def plot_convergence(ax_or_fig, convergence, target_chi2=1.0):
    """수렴 진단 다중 패널 (학술 논문용).

    Parameters
    ----------
    ax_or_fig : matplotlib Figure 또는 단일 Axes
    convergence : dict (Inversion2D.convergence)
    target_chi2 : float, 목표 chi²/N
    """
    if hasattr(ax_or_fig, 'add_subplot'):
        fig = ax_or_fig
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    else:
        return

    iters = np.arange(1, len(convergence['rms']) + 1)
    if len(iters) == 0:
        return

    # (a) RMS vs iteration
    ax1.semilogy(iters, convergence['rms'], 'bo-', ms=5, lw=1.5)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('RMS (log domain)')
    ax1.set_title('(a) Data Misfit (RMS)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # (b) chi²/N vs iteration (+ target line)
    ax2.plot(iters, convergence['chi2'], 'rs-', ms=5, lw=1.5)
    ax2.axhline(target_chi2, color='green', ls='--', lw=1.5,
                label=f'Target χ²/N = {target_chi2:.1f}')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('χ²/N')
    ax2.set_title('(b) Normalized Chi-squared', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # (c) Alpha vs iteration
    ax3.semilogy(iters, convergence['alpha'], 'g^-', ms=5, lw=1.5)
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('α (regularization)')
    ax3.set_title('(c) Regularization Parameter', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # (d) L-curve: φ_d vs φ_m
    ax4.loglog(convergence['phi_d'], convergence['phi_m'], 'kD-', ms=5, lw=1.5)
    for i, (pd, pm) in enumerate(zip(convergence['phi_d'], convergence['phi_m'])):
        ax4.annotate(str(i+1), (pd, pm), fontsize=7, ha='left', va='bottom')
    ax4.set_xlabel('φ_d (data misfit)'); ax4.set_ylabel('φ_m (model roughness)')
    ax4.set_title('(d) Trade-off Curve', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)


def plot_sensitivity_section(ax, mesh, sensitivity, survey,
                             title="Cumulative Sensitivity",
                             inv_blocks=None, clip_electrodes=5):
    """누적 감도 단면도 (Friedel, 2003).

    Parameters
    ----------
    sensitivity : array, 정규화된 감도 (0~1), 블록 또는 셀 기준
    """
    a = survey.a; ex = survey.electrode_x
    z_max = survey.n_max * a * 1.25

    if inv_blocks is not None:
        src_x = np.array([b['cx'] for b in inv_blocks.blocks])
        src_z = np.array([b['cz'] for b in inv_blocks.blocks])
        src_val = sensitivity[:len(inv_blocks.blocks)]
    else:
        ncx, ncz = mesh.ncx, mesh.ncz
        CX, CZ = np.meshgrid(mesh.x_cc, mesh.z_cc)
        src_x = CX.ravel()[:len(sensitivity)]
        src_z = CZ.ravel()[:len(sensitivity)]
        src_val = sensitivity

    xi = np.linspace(ex[0] - a, ex[-1] + a, 400)
    zi = np.linspace(0, z_max, 200)
    XI, ZI = np.meshgrid(xi, zi)
    VI = griddata((src_x, src_z), src_val, (XI, ZI), method='cubic')
    VI_fill = griddata((src_x, src_z), src_val, (XI, ZI), method='nearest')
    VI[np.isnan(VI)] = VI_fill[np.isnan(VI)]

    # 사다리꼴 마스크
    n_cut = max(0, int(clip_electrodes))
    if n_cut > 0 and n_cut < len(ex) // 2:
        x_left = ex[n_cut]; x_right = ex[-(n_cut + 1)]
    else:
        x_left = ex[0]; x_right = ex[-1]
    slope = 0.7
    left_b = x_left - ZI * slope
    right_b = x_right + ZI * slope
    # 전체 범위에서 경사
    out_mask = (XI < (ex[0] + ZI * slope)) | (XI > (ex[-1] - ZI * slope))
    VI[out_mask] = np.nan

    im = ax.pcolormesh(XI, ZI, VI, cmap='inferno', vmin=0, vmax=1,
                       shading='gouraud', rasterized=True)
    # 등치선
    try:
        cs = ax.contour(XI, ZI, VI, levels=[0.2, 0.4, 0.6, 0.8],
                        colors='white', linewidths=0.8, zorder=12)
        ax.clabel(cs, fmt='%.1f', fontsize=7, inline=True)
    except Exception:
        pass

    for i, xe in enumerate(ex):
        ax.plot(xe, 0, 'wv', markersize=3, zorder=10)
    ax.set_xlim(ex[0] - a * 0.5, ex[-1] + a * 0.5)
    ax.set_ylim(z_max + a * 0.3, -a * 0.6)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Depth (m)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.45)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Normalized Sensitivity (0=low, 1=high)')
    return im


def plot_resolution_section(ax, mesh, resolution, survey,
                            title="Model Resolution",
                            inv_blocks=None, clip_electrodes=5):
    """모델 분해능 대각 성분 단면도 (Menke, 2012)."""
    a = survey.a; ex = survey.electrode_x
    z_max = survey.n_max * a * 1.25

    if inv_blocks is not None:
        src_x = np.array([b['cx'] for b in inv_blocks.blocks])
        src_z = np.array([b['cz'] for b in inv_blocks.blocks])
        src_val = resolution[:len(inv_blocks.blocks)]
    else:
        ncx, ncz = mesh.ncx, mesh.ncz
        CX, CZ = np.meshgrid(mesh.x_cc, mesh.z_cc)
        src_x = CX.ravel()[:len(resolution)]
        src_z = CZ.ravel()[:len(resolution)]
        src_val = resolution

    xi = np.linspace(ex[0] - a, ex[-1] + a, 400)
    zi = np.linspace(0, z_max, 200)
    XI, ZI = np.meshgrid(xi, zi)
    VI = griddata((src_x, src_z), src_val, (XI, ZI), method='cubic')
    VI_fill = griddata((src_x, src_z), src_val, (XI, ZI), method='nearest')
    VI[np.isnan(VI)] = VI_fill[np.isnan(VI)]

    out_mask = (XI < (ex[0] + ZI * 0.7)) | (XI > (ex[-1] - ZI * 0.7))
    VI[out_mask] = np.nan

    im = ax.pcolormesh(XI, ZI, VI, cmap='YlOrRd', vmin=0, vmax=1,
                       shading='gouraud', rasterized=True)
    try:
        cs = ax.contour(XI, ZI, VI, levels=[0.1, 0.3, 0.5, 0.7],
                        colors='black', linewidths=0.8, zorder=12)
        ax.clabel(cs, fmt='%.1f', fontsize=7, inline=True)
    except Exception:
        pass

    for i, xe in enumerate(ex):
        ax.plot(xe, 0, 'kv', markersize=3, zorder=10)
    ax.set_xlim(ex[0] - a * 0.5, ex[-1] + a * 0.5)
    ax.set_ylim(z_max + a * 0.3, -a * 0.6)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Depth (m)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.45)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Resolution (0=unresolved, 1=perfectly resolved)')
    return im


# ============================================================
# GUI 다국어 사전 (KO → EN) — 위젯 라벨 토글용
# ============================================================
KO_EN = {
    '데이터 입력': 'Data Input',
    '탐사 설정': 'Survey Settings',
    '전극 간격 a (m):': 'Electrode spacing a (m):',
    '전극 수:': 'No. of electrodes:',
    '최대 n-level:': 'Max n-level:',
    '지역명:': 'Site name:',
    '측선명:': 'Line name:',
    '데이터 유형:': 'Data type:',
    '테이블 생성': 'Create table',
    'APV 저장': 'Save APV',
    'RES2DINV 저장': 'Save RES2DINV',
    '→ 역산으로 전송': '→ Send to inversion',
    '테이블을 생성하세요': 'Create a table',
    '데이터 불러오기': 'Load Data',
    'APV 파일 & 탐사 정보': 'APV file & survey info',
    'APV 불러오기': 'Load APV',
    'RES2DINV 불러오기': 'Load RES2DINV',
    '불량 데이터 자동 제거 (음수/이상치)': 'Auto-remove bad data (negative/outliers)',
    '전극 표고 (쉼표 구분, m):': 'Electrode elevation (comma-sep, m):',
    '예: 100,99,98,97,...  (비우면 평탄 지형)': 'e.g. 100,99,98,97,...  (empty = flat terrain)',
    '의사단면도 표시': 'Show pseudosection',
    '→ 역산으로 이동': '→ Go to inversion',
    '역산': 'Inversion',
    '역산 파라미터': 'Inversion parameters',
    "'auto': 관측 평균값 사용": "'auto': use mean of observed data",
    '전방 모델링:': 'Forward modeling:',
    '역산 격자:': 'Inversion grid:',
    '블록': 'Block',
    '셀(세밀)': 'Cell (fine)',
    '정규화:': 'Regularization:',
    'L2(평활)': 'L2 (smooth)',
    'MGS(경계)': 'MGS (sharp)',
    '경사 평활 (°):': 'Dip smoothing (°):',
    '강도:': 'Strength:',
    'L-curve 자동 정규화': 'L-curve auto regularization',
    'Robust 역산': 'Robust inversion',
    '모델 블록 격자선 표시': 'Show model block grid',
    '단면도에 비저항 값 표시': 'Show resistivity values on section',
    '컬러 범위 (Ω·m):': 'Color range (Ω·m):',
    '양끝 전극 제거 (4개)': 'Trim end electrodes (4)',
    '등치선 (Ω·m):': 'Contours (Ω·m):',
    '쉼표 구분 (비우면 없음)': 'Comma-separated (empty = none)',
    '데이터 오차 모델:': 'Data error model:',
    '상대 오차 (%):': 'Relative error (%):',
    '노이즈 하한:': 'Noise floor:',
    '목표 χ²/N:': 'Target χ²/N:',
    '역산 실행': 'Run inversion',
    '다시 그리기': 'Redraw',
    '전체 저장': 'Save all',
    '단면도만 저장': 'Save section only',
    '학술 분석:': 'Academic analysis:',
    '수렴 곡선': 'Convergence curve',
    '감도 분석': 'Sensitivity analysis',
    '분해능 분석': 'Resolution analysis',
    '결과 내보내기': 'Export results',
    'DOI 신뢰도 분석': 'DOI reliability analysis',
    '전방 모델링': 'Forward Modeling',
    '모델 설정': 'Model settings',
    '배경 비저항 (Ω·m):': 'Background resistivity (Ω·m):',
    '이상체 (x1 z1 x2 z2 ρ):': 'Anomaly (x1 z1 x2 z2 ρ):',
    '추가': 'Add',
    '삭제': 'Delete',
    '마우스로 모델 그리기': 'Draw model with mouse',
    '드래그: 영역 선택 → 비저항 입력': 'Drag: select region → enter resistivity',
    '모델 미리보기': 'Model preview',
    '전방 모델링 실행': 'Run forward modeling',
    '→ 역산 데이터로 전송 (3% 노이즈)': '→ Send as inversion data (3% noise)',
}


# ============================================================
# GUI 애플리케이션
# ============================================================
class DipoleDipoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RESIS Pro - 2D Resistivity Inversion System")
        # 화면 크기에 맞춤
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        try:
            self.root.state('zoomed')  # Windows
        except:
            pass  # macOS는 위 geometry로 충분
        self.survey = None; self.mesh = None
        self.obs_data = None; self.apv_info = None

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('맑은 고딕', 11, 'bold'))

        # 언어 토글 바 (KO/EN)
        self.lang = 'KO'
        topbar = ttk.Frame(root)
        topbar.pack(fill='x', side='top', padx=8, pady=(4, 0))
        ttk.Label(topbar, text="Language / 언어:").pack(side='right', padx=(0, 4))
        self.lang_var = tk.StringVar(value='한국어')
        lang_cb = ttk.Combobox(topbar, textvariable=self.lang_var, width=10,
                               state='readonly', values=['한국어', 'English'])
        lang_cb.pack(side='right')
        lang_cb.bind('<<ComboboxSelected>>', lambda e: self._apply_language(
            'EN' if self.lang_var.get() == 'English' else 'KO'))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self._create_input_tab()
        self._create_data_tab()
        self._create_inversion_tab()
        self._create_forward_tab()

        self.status_var = tk.StringVar(value="준비 - APV 파일을 불러오세요")
        ttk.Label(root, textvariable=self.status_var,
                  relief='sunken', anchor='w', font=('맑은 고딕', 10)).pack(fill='x', side='bottom')

    def _get_topography(self):
        """지형 입력 파싱. 없으면 None 반환."""
        txt = self.topo_entry.get().strip()
        if not txt:
            return None
        try:
            topo = np.array([float(v) for v in txt.replace(' ', ',').split(',') if v])
            return topo if len(topo) >= 2 else None
        except ValueError:
            return None

    def _make_mesh(self, survey):
        """지형 정보 포함 Mesh 생성"""
        return Mesh2D(survey, topography=self._get_topography())

    # ── 다국어 토글 (위젯 라벨 KO↔EN) ──
    def _collect_widgets(self, w, acc):
        acc.append(w)
        try:
            for c in w.winfo_children():
                self._collect_widgets(c, acc)
        except Exception:
            pass

    def _tr_en(self, s):
        """한국어 라벨 → 영어 (앞뒤 공백 보존). 사전에 없으면 원문 유지."""
        if s in KO_EN:
            return KO_EN[s]
        st = s.strip()
        if st in KO_EN:
            lead = s[:len(s) - len(s.lstrip())]
            trail = s[len(s.rstrip()):]
            return lead + KO_EN[st] + trail
        return s

    def _apply_language(self, lang):
        """모든 위젯 텍스트와 탭 라벨을 lang('KO'/'EN')으로 전환."""
        self.lang = lang
        if not hasattr(self, '_orig_texts'):
            self._orig_texts = {}
        if not hasattr(self, '_orig_tabs'):
            self._orig_tabs = {}
        widgets = []
        self._collect_widgets(self.root, widgets)
        for w in widgets:
            try:
                t = w.cget('text')
            except Exception:
                continue
            if not isinstance(t, str) or not t:
                continue
            wid = str(w)
            if wid not in self._orig_texts:
                self._orig_texts[wid] = t   # 최초 1회 한국어 원문 캐시
            orig = self._orig_texts[wid]
            try:
                w.configure(text=(self._tr_en(orig) if lang == 'EN' else orig))
            except Exception:
                pass
        # 노트북 탭 라벨
        try:
            for i in range(self.notebook.index('end')):
                if i not in self._orig_tabs:
                    self._orig_tabs[i] = self.notebook.tab(i, 'text')
                orig = self._orig_tabs[i]
                self.notebook.tab(i, text=(self._tr_en(orig) if lang == 'EN' else orig))
        except Exception:
            pass
        # 창 제목
        self.root.title("RESIS Pro - 2D Resistivity Inversion System" if lang == 'EN'
                        else "RESIS Pro - 2D 비저항 탐사 역산 시스템")

    # ── Tab 0: 데이터 직접 입력 ──
    def _create_input_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  데이터 입력  ")

        # 좌측: 탐사 설정
        ctrl = ttk.LabelFrame(tab, text="탐사 설정")
        ctrl.pack(side='left', fill='y', padx=5, pady=5)

        ttk.Label(ctrl, text="전극 간격 a (m):").grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.inp_a = ttk.Entry(ctrl, width=8); self.inp_a.insert(0, "5")
        self.inp_a.grid(row=0, column=1, padx=5)
        ttk.Label(ctrl, text="전극 수:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.inp_ne = ttk.Entry(ctrl, width=8); self.inp_ne.insert(0, "29")
        self.inp_ne.grid(row=1, column=1, padx=5)
        ttk.Label(ctrl, text="최대 n-level:").grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.inp_nm = ttk.Entry(ctrl, width=8); self.inp_nm.insert(0, "4")
        self.inp_nm.grid(row=2, column=1, padx=5)
        ttk.Label(ctrl, text="지역명:").grid(row=3, column=0, sticky='w', padx=5, pady=3)
        self.inp_area = ttk.Entry(ctrl, width=12); self.inp_area.insert(0, "현장명")
        self.inp_area.grid(row=3, column=1, padx=5)
        ttk.Label(ctrl, text="측선명:").grid(row=4, column=0, sticky='w', padx=5, pady=3)
        self.inp_line = ttk.Entry(ctrl, width=12); self.inp_line.insert(0, "LINE-1")
        self.inp_line.grid(row=4, column=1, padx=5)

        ttk.Label(ctrl, text="데이터 유형:").grid(row=5, column=0, sticky='w', padx=5, pady=3)
        self.inp_dtype = tk.StringVar(value='V_over_I')
        ttk.Radiobutton(ctrl, text="V/I", variable=self.inp_dtype,
                        value='V_over_I').grid(row=5, column=1, sticky='w')
        ttk.Radiobutton(ctrl, text="ρa", variable=self.inp_dtype,
                        value='rho_a').grid(row=6, column=1, sticky='w')

        ttk.Separator(ctrl).grid(row=7, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Button(ctrl, text="테이블 생성",
                   command=self._create_input_table).grid(row=8, column=0, columnspan=2, pady=5)

        save_frame = ttk.Frame(ctrl)
        save_frame.grid(row=9, column=0, columnspan=2, pady=3)
        ttk.Button(save_frame, text="APV 저장",
                   command=self._save_input_apv).pack(side='left', padx=2)
        ttk.Button(save_frame, text="RES2DINV 저장",
                   command=self._save_input_res2dinv).pack(side='left', padx=2)

        ttk.Button(ctrl, text="→ 역산으로 전송",
                   command=self._send_input_to_inv).grid(row=10, column=0, columnspan=2, pady=3)

        ttk.Separator(ctrl).grid(row=11, column=0, columnspan=2, sticky='ew', pady=5)
        self.inp_status = ttk.Label(ctrl, text="테이블을 생성하세요", font=('맑은 고딕', 9))
        self.inp_status.grid(row=12, column=0, columnspan=2, padx=5, pady=5)

        # 우측: n-level별 데이터 테이블
        table_frame = ttk.Frame(tab)
        table_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        # 스크롤 가능한 테이블 영역
        canvas = tk.Canvas(table_frame)
        v_sb = ttk.Scrollbar(table_frame, orient='vertical', command=canvas.yview)
        h_sb = ttk.Scrollbar(table_frame, orient='horizontal', command=canvas.xview)
        canvas.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        v_sb.pack(side='right', fill='y')
        h_sb.pack(side='bottom', fill='x')
        canvas.pack(side='left', fill='both', expand=True)

        self.inp_inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.inp_inner, anchor='nw')
        self.inp_inner.bind('<Configure>',
                            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        self.inp_canvas = canvas
        self.inp_entries = {}  # {(n, j): Entry widget}

    def _create_input_table(self):
        """의사단면도 배치로 n-level별 입력 테이블 생성

        각 셀이 실제 측정 위치(x 중심점)에 맞게 배치됨.
        n=1: 왼쪽 정렬, n=2: 오른쪽으로 0.5칸 이동, ...
        → 사다리꼴 형태로 직관적 입력
        """
        try:
            a = float(self.inp_a.get())
            ne = int(self.inp_ne.get())
            nm = int(self.inp_nm.get())
        except ValueError:
            messagebox.showerror("오류", "파라미터를 확인하세요."); return

        for w in self.inp_inner.winfo_children():
            w.destroy()
        self.inp_entries = {}

        # 전극 번호 헤더 (2칸 단위)
        # 열 구조: col 0 = n-level 라벨, col 1~ = 반전극간격(a/2) 단위 배치
        # 측정 j, n-level k의 midpoint 열: 2*j + k + 2
        max_col = 2 * (ne - 3) + nm + 2

        # 전극 위치 표시 (상단)
        for i in range(ne):
            col = 2 * i + 1
            if col <= max_col:
                ttk.Label(self.inp_inner, text=f'▼{i+1}', font=('맑은 고딕', 7),
                          foreground='gray').grid(row=0, column=col, padx=0)

        for n in range(1, nm + 1):
            n_data = ne - n - 2
            if n_data <= 0:
                continue

            r = n  # 행 번호
            # n-level 라벨
            ttk.Label(self.inp_inner, text=f'n={n}',
                      font=('맑은 고딕', 9, 'bold')).grid(row=r, column=0, padx=3, pady=1)

            for j in range(n_data):
                # 측정 midpoint 열: (2j + n + 2) → 의사단면도 x 위치에 대응
                col = 2 * j + n + 2
                e = ttk.Entry(self.inp_inner, width=6, justify='center',
                              font=('맑은 고딕', 8))
                e.grid(row=r, column=col, padx=0, pady=1)
                self.inp_entries[(n, j)] = e

        total = sum(ne - n - 2 for n in range(1, nm + 1) if ne - n - 2 > 0)
        self.inp_status.config(text=f"테이블 생성: {nm}행, 총 {total}개 입력칸")

    def _get_input_data(self):
        """테이블에서 데이터 읽기 → data_by_n dict"""
        nm = int(self.inp_nm.get())
        ne = int(self.inp_ne.get())
        data_by_n = {}
        for n in range(1, nm + 1):
            n_data = ne - n - 2
            if n_data <= 0: continue
            vals = []
            for j in range(n_data):
                e = self.inp_entries.get((n, j))
                if e:
                    txt = e.get().strip()
                    try:
                        vals.append(float(txt))
                    except ValueError:
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            data_by_n[n] = vals
        return data_by_n

    def _save_input_apv(self):
        """입력 데이터를 APV 파일로 저장"""
        data_by_n = self._get_input_data()
        if not data_by_n:
            messagebox.showerror("오류", "먼저 테이블을 생성하고 데이터를 입력하세요."); return

        a = float(self.inp_a.get())
        ne = int(self.inp_ne.get())
        nm = int(self.inp_nm.get())
        area = self.inp_area.get().strip()
        line = self.inp_line.get().strip()

        fp = filedialog.asksaveasfilename(
            title="APV 파일 저장",
            defaultextension='.APV',
            filetypes=[("APV 파일", "*.APV *.apv"), ("모든 파일", "*.*")])
        if not fp: return

        with open(fp, 'w') as f:
            f.write('V4\n         1\n   0\n')
            f.write(f'{area}\n{line}\n')
            n1_count = len(data_by_n.get(1, []))
            f.write(f'   {nm}  {n1_count}   0\n')
            for n in range(1, nm + 1):
                vals = data_by_n.get(n, [])
                for j in range(0, len(vals), 10):
                    chunk = vals[j:j + 10]
                    f.write('  ' + '  '.join(f'{v:.5f}' for v in chunk) + ' \n')
            f.write(f'    1    1\nTEST\n   {a:.3f}\nDIPRO data\n\n')
            for i in range(1, ne):
                f.write(f'{i}\n')

        self.inp_status.config(text=f"저장 완료: {os.path.basename(fp)}")
        self.status_var.set(f"APV 저장: {fp}")

    def _save_input_res2dinv(self):
        """입력 데이터를 RES2DINV .dat 형식으로 저장

        RES2DINV 형식:
        Line 1: 측선명
        Line 2: 전극 간격
        Line 3: 배열 유형 (3=dipole-dipole)
        Line 4: 총 데이터 수
        Line 5: 위치 유형 (1=첫 전극 위치)
        Line 6: 플래그 (0)
        Data: x_c1  a  n  rho_a (각 측정당 1행)
        마지막: 0 0 0 0
        """
        data_by_n = self._get_input_data()
        if not data_by_n:
            messagebox.showerror("오류", "먼저 데이터를 입력하세요."); return

        a = float(self.inp_a.get())
        ne = int(self.inp_ne.get())
        nm = int(self.inp_nm.get())
        dt = self.inp_dtype.get()
        line_name = self.inp_line.get().strip()

        fp = filedialog.asksaveasfilename(
            title="RES2DINV 파일 저장",
            defaultextension='.dat',
            filetypes=[("RES2DINV 파일", "*.dat"), ("모든 파일", "*.*")])
        if not fp: return

        # 겉보기 비저항 계산
        electrode_x = np.arange(ne) * a
        total_data = 0
        data_lines = []
        for n in range(1, nm + 1):
            vals = data_by_n.get(n, [])
            K = np.pi * a * n * (n + 1) * (n + 2)
            for j in range(len(vals)):
                if vals[j] == 0: continue
                x_c1 = electrode_x[j]  # 첫 번째 전류 전극 위치
                if dt == 'V_over_I':
                    rho_a = K * vals[j]
                else:
                    rho_a = vals[j]
                data_lines.append(f"  {x_c1:.2f}  {a:.2f}  {n}  {rho_a:.4f}\n")
                total_data += 1

        with open(fp, 'w') as f:
            f.write(f"{line_name}\n")
            f.write(f"{a:.2f}\n")
            f.write("3\n")  # 3 = dipole-dipole
            f.write(f"{total_data}\n")
            f.write("1\n")  # 1 = first electrode location
            f.write("0\n")
            for line in data_lines:
                f.write(line)
            f.write("0  0  0  0\n")
            f.write("0  0  0  0  0\n")
            f.write("0\n")

        self.inp_status.config(text=f"RES2DINV 저장: {os.path.basename(fp)}")
        self.status_var.set(f"RES2DINV 저장: {fp} ({total_data}개 데이터)")

    def _send_input_to_inv(self):
        """입력 데이터를 역산 탭으로 전송"""
        data_by_n = self._get_input_data()
        if not data_by_n:
            messagebox.showerror("오류", "먼저 데이터를 입력하세요."); return

        a = float(self.inp_a.get())
        ne = int(self.inp_ne.get())
        nm = int(self.inp_nm.get())
        dt = self.inp_dtype.get()

        # APV info 형식으로 변환
        info = {
            'a': a, 'n_max': nm, 'n_electrodes': ne + 3,  # parse_apv 호환
            'area': self.inp_area.get(), 'line': self.inp_line.get(),
            'data_by_n': {n: np.array(v) for n, v in data_by_n.items()},
        }
        # n_electrodes 재계산: n=1 측정수 = ne - 3
        n1_count = len(data_by_n.get(1, []))
        info['n_electrodes'] = n1_count + 3
        info['n1_count'] = n1_count

        elec_x, meas, rho_a = apv_to_survey_data(info, dt)
        self.survey = DipDipSurvey(
            a=a, n_electrodes=info['n_electrodes'],
            n_max=nm, electrode_x=elec_x, measurements=meas)
        self.obs_data = rho_a

        if hasattr(self, 'filter_var') and self.filter_var.get():
            self.survey, self.obs_data, _ = filter_bad_data(
                self.survey, self.obs_data, verbose=False)

        self.mesh = self._make_mesh(self.survey)
        self.notebook.select(2)  # 역산 탭으로 이동
        self.inp_status.config(text=f"전송 완료: {self.survey.n_data}개 측정")
        self.status_var.set(f"데이터 전송 완료: {self.survey.n_data}개 → 역산 탭")

    # ── Tab 1: 데이터 불러오기 ──
    def _create_data_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  데이터 불러오기  ")

        ctrl = ttk.LabelFrame(tab, text="APV 파일 & 탐사 정보")
        ctrl.pack(side='left', fill='y', padx=5, pady=5)

        load_frame = ttk.Frame(ctrl)
        load_frame.pack(padx=10, pady=5, fill='x')
        ttk.Button(load_frame, text="APV 불러오기",
                   command=self._load_apv).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(load_frame, text="RES2DINV 불러오기",
                   command=self._load_res2dinv).pack(side='left', expand=True, fill='x', padx=2)

        ttk.Separator(ctrl).pack(fill='x', pady=5)

        # 데이터 유형 선택
        ttk.Label(ctrl, text="데이터 유형:").pack(anchor='w', padx=10)
        self.data_type_var = tk.StringVar(value='V_over_I')
        for text, val in [("전위/전류 (V/I = 저항값)", 'V_over_I'),
                          ("겉보기 비저항 (ρa)", 'rho_a')]:
            ttk.Radiobutton(ctrl, text=text, variable=self.data_type_var,
                            value=val).pack(anchor='w', padx=20)

        ttk.Separator(ctrl).pack(fill='x', pady=5)

        # 불량 데이터 필터링 옵션
        self.filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="불량 데이터 자동 제거 (음수/이상치)",
                        variable=self.filter_var).pack(anchor='w', padx=10, pady=3)

        ttk.Separator(ctrl).pack(fill='x', pady=5)

        # 지형 입력
        ttk.Label(ctrl, text="전극 표고 (쉼표 구분, m):").pack(anchor='w', padx=10)
        self.topo_entry = ttk.Entry(ctrl, width=28)
        self.topo_entry.pack(padx=10, pady=2, fill='x')
        ttk.Label(ctrl, text="  예: 100,99,98,97,...  (비우면 평탄 지형)",
                  font=('맑은 고딕', 8)).pack(anchor='w', padx=10)

        ttk.Separator(ctrl).pack(fill='x', pady=5)

        # 탐사 정보 표시
        self.data_info = tk.Text(ctrl, width=28, height=15, state='disabled',
                                  font=('Courier', 10))
        self.data_info.pack(padx=10, pady=5, fill='both', expand=True)

        ttk.Button(ctrl, text="의사단면도 표시",
                   command=self._show_pseudosection).pack(padx=10, pady=5, fill='x')
        ttk.Button(ctrl, text="→ 역산으로 이동",
                   command=lambda: self.notebook.select(1)).pack(padx=10, pady=5, fill='x')

        # Plot
        self.data_fig = Figure(figsize=(12, 7))
        canvas = FigureCanvasTkAgg(self.data_fig, tab)
        toolbar = NavigationToolbar2Tk(canvas, tab); toolbar.update()
        canvas.get_tk_widget().pack(side='right', fill='both', expand=True)
        self.data_canvas = canvas

    def _load_apv(self):
        fp = filedialog.askopenfilename(
            title="APV 파일 선택",
            filetypes=[("APV 파일", "*.APV *.apv"), ("모든 파일", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)))
        if not fp: return
        try:
            self.apv_info = parse_apv(fp)
            info = self.apv_info
            dt = self.data_type_var.get()
            elec_x, meas, rho_a = apv_to_survey_data(info, dt)

            self.survey = DipDipSurvey(
                a=info['a'], n_electrodes=info['n_electrodes'],
                n_max=info['n_max'], electrode_x=elec_x, measurements=meas)
            self.obs_data = rho_a

            # 불량 데이터 필터링
            filter_report = ""
            if self.filter_var.get():
                self.survey, self.obs_data, filter_report = \
                    filter_bad_data(self.survey, self.obs_data, verbose=False)

            self.mesh = self._make_mesh(self.survey)

            # 정보 표시
            txt = (f"파일: {os.path.basename(fp)}\n"
                   f"지역: {info['area']}\n"
                   f"측선: {info['line']}\n"
                   f"────────────────────\n"
                   f"전극 간격 (a): {info['a']:.1f} m\n"
                   f"전극 수: {info['n_electrodes']}\n"
                   f"측선 길이: {elec_x[-1]:.1f} m\n"
                   f"최대 n-level: {info['n_max']}\n"
                   f"────────────────────\n"
                   f"원본 측정 수: {len(rho_a)}\n"
                   f"{filter_report}\n"
                   f"사용 측정 수: {self.survey.n_data}\n")
            for n in range(1, info['n_max'] + 1):
                d = info['data_by_n'][n]
                K = np.pi * info['a'] * n * (n + 1) * (n + 2)
                if dt == 'V_over_I':
                    ra = d * K
                else:
                    ra = d
                txt += f"  n={n}: {len(d)}개, ρa={ra.min():.0f}~{ra.max():.0f} Ω·m\n"
            txt += (f"────────────────────\n"
                    f"격자: {self.mesh.ncx}×{self.mesh.ncz} = {self.mesh.n_cells} 셀\n"
                    f"겉보기 비저항 범위:\n"
                    f"  {rho_a.min():.1f} ~ {rho_a.max():.1f} Ω·m\n"
                    f"  평균: {rho_a.mean():.1f} Ω·m\n")

            self.data_info.config(state='normal')
            self.data_info.delete('1.0', 'end')
            self.data_info.insert('1.0', txt)
            self.data_info.config(state='disabled')

            self._show_pseudosection()
            self.status_var.set(f"데이터 로드 완료: {os.path.basename(fp)} ({self.survey.n_data}개 측정)")

        except Exception as e:
            messagebox.showerror("오류", f"파일 읽기 실패:\n{e}")

    def _load_res2dinv(self):
        """RES2DINV .dat 파일 불러오기"""
        fp = filedialog.askopenfilename(
            title="RES2DINV 파일 선택",
            filetypes=[("RES2DINV 파일", "*.dat *.DAT"), ("모든 파일", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)))
        if not fp: return
        try:
            res = parse_res2dinv(fp)

            self.survey = DipDipSurvey(
                a=res['a'], n_electrodes=res['n_electrodes'],
                n_max=res['n_max'], electrode_x=res['electrode_x'],
                measurements=res['measurements'],
                array_type=res['array_type'])
            self.obs_data = res['rho_a']
            self.apv_info = None  # APV가 아닌 RES2DINV 소스

            if hasattr(self, 'filter_var') and self.filter_var.get():
                self.survey, self.obs_data, filter_report = \
                    filter_bad_data(self.survey, self.obs_data, verbose=False)
            else:
                filter_report = ""

            self.mesh = self._make_mesh(self.survey)

            txt = (f"파일: {os.path.basename(fp)}\n"
                   f"형식: RES2DINV .dat\n"
                   f"측선: {res['line']}\n"
                   f"배열: {res['array_type']}\n"
                   f"────────────────────\n"
                   f"전극 간격: {res['a']:.1f} m\n"
                   f"전극 수: {res['n_electrodes']}\n"
                   f"최대 n-level: {res['n_max']}\n"
                   f"────────────────────\n"
                   f"원본 측정 수: {len(res['rho_a'])}\n"
                   f"{filter_report}\n"
                   f"사용 측정 수: {self.survey.n_data}\n"
                   f"ρa 범위: {self.obs_data.min():.1f}~{self.obs_data.max():.1f} Ω·m\n")

            self.data_info.config(state='normal')
            self.data_info.delete('1.0', 'end')
            self.data_info.insert('1.0', txt)
            self.data_info.config(state='disabled')

            self._show_pseudosection()
            self.status_var.set(f"RES2DINV 로드: {os.path.basename(fp)} ({self.survey.n_data}개)")
        except Exception as e:
            messagebox.showerror("오류", f"파일 읽기 실패:\n{e}")

    def _show_pseudosection(self):
        if self.obs_data is None:
            messagebox.showinfo("알림", "먼저 APV 파일을 불러오세요.")
            return
        if self.apv_info:
            dt = self.data_type_var.get()
            elec_x, meas, rho_a = apv_to_survey_data(self.apv_info, dt)
            self.survey = DipDipSurvey(
                a=self.apv_info['a'], n_electrodes=self.apv_info['n_electrodes'],
                n_max=self.apv_info['n_max'], electrode_x=elec_x, measurements=meas)
            self.obs_data = rho_a
            self.mesh = self._make_mesh(self.survey)

        fig = self.data_fig; fig.clear()
        ax = fig.add_subplot(111)
        plot_pseudosection(ax, self.survey, self.obs_data,
                           "겉보기 비저항 의사단면도")
        fig.tight_layout()
        self.data_canvas.draw()

    # ── Tab 2: 역산 ──
    def _create_inversion_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  역산  ")

        # ── 스크롤 가능한 왼쪽 패널 ──
        ctrl_outer = ttk.LabelFrame(tab, text="역산 파라미터")
        ctrl_outer.pack(side='left', fill='y', padx=5, pady=5)

        ctrl_canvas = tk.Canvas(ctrl_outer, width=260, highlightthickness=0)
        ctrl_scrollbar = ttk.Scrollbar(ctrl_outer, orient='vertical',
                                        command=ctrl_canvas.yview)
        ctrl = ttk.Frame(ctrl_canvas)

        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)

        ctrl_scrollbar.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)

        # 마우스 휠 스크롤 지원
        def _on_mousewheel(event):
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_linux(event):
            if event.num == 4:
                ctrl_canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                ctrl_canvas.yview_scroll(3, "units")
        ctrl_canvas.bind('<MouseWheel>', _on_mousewheel)          # Windows/Mac
        ctrl_canvas.bind('<Button-4>', _on_mousewheel_linux)      # Linux up
        ctrl_canvas.bind('<Button-5>', _on_mousewheel_linux)      # Linux down
        # 내부 위젯 위에서도 스크롤 동작
        def _bind_mousewheel(widget):
            widget.bind('<MouseWheel>', _on_mousewheel)
            widget.bind('<Button-4>', _on_mousewheel_linux)
            widget.bind('<Button-5>', _on_mousewheel_linux)
            for child in widget.winfo_children():
                _bind_mousewheel(child)
        ctrl.bind('<Map>', lambda e: _bind_mousewheel(ctrl))

        params = [
            ("기준 비저항 (Ω·m):", "auto", "inv_ref"),
            ("정규화 α:", "1.0", "inv_alpha"),
            ("x-평활:", "1.0", "inv_ax"),
            ("z-평활:", "1.0", "inv_az"),
            ("최대 반복:", "8", "inv_iter"),
            ("수렴 기준 RMS:", "0.05", "inv_tol"),
        ]
        self.inv_entries = {}
        for i, (lbl, val, key) in enumerate(params):
            ttk.Label(ctrl, text=lbl).grid(row=i, column=0, sticky='w', padx=5, pady=4)
            e = ttk.Entry(ctrl, width=10); e.insert(0, val)
            e.grid(row=i, column=1, padx=5, pady=4)
            self.inv_entries[key] = e

        ttk.Label(ctrl, text="'auto': 관측 평균값 사용",
                  font=('맑은 고딕', 8)).grid(row=len(params), column=0, columnspan=2, sticky='w', padx=5)

        ttk.Separator(ctrl).grid(row=len(params) + 1, column=0, columnspan=2, sticky='ew', pady=5)

        # 전방 모델링 방법 선택
        ttk.Label(ctrl, text="전방 모델링:").grid(row=len(params) + 2, column=0, sticky='w', padx=10)
        self.inv_solver_var = tk.StringVar(value='FDM')
        sf = ttk.Frame(ctrl)
        sf.grid(row=len(params) + 2, column=1, sticky='w')
        ttk.Radiobutton(sf, text="FDM", variable=self.inv_solver_var, value='FDM').pack(side='left')
        ttk.Radiobutton(sf, text="FEM", variable=self.inv_solver_var, value='FEM').pack(side='left')

        # 파라미터화 선택
        ttk.Label(ctrl, text="역산 격자:").grid(row=len(params) + 3, column=0, sticky='w', padx=10)
        self.inv_param_var = tk.StringVar(value='block')
        pf = ttk.Frame(ctrl)
        pf.grid(row=len(params) + 3, column=1, sticky='w')
        ttk.Radiobutton(pf, text="블록", variable=self.inv_param_var, value='block').pack(side='left')
        ttk.Radiobutton(pf, text="셀(세밀)", variable=self.inv_param_var, value='cell').pack(side='left')

        # 정규화 선택
        ttk.Label(ctrl, text="정규화:").grid(row=len(params) + 4, column=0, sticky='w', padx=10)
        self.inv_reg_var = tk.StringVar(value='L2')
        rf = ttk.Frame(ctrl)
        rf.grid(row=len(params) + 4, column=1, sticky='w')
        ttk.Radiobutton(rf, text="L2(평활)", variable=self.inv_reg_var, value='L2').pack(side='left')
        ttk.Radiobutton(rf, text="MGS(경계)", variable=self.inv_reg_var, value='MGS').pack(side='left')

        # 이방성 정규화
        ttk.Label(ctrl, text="경사 평활 (°):").grid(
            row=len(params) + 5, column=0, sticky='w', padx=10)
        dip_frame = ttk.Frame(ctrl)
        dip_frame.grid(row=len(params) + 5, column=1, sticky='w')
        self.dip_angle_entry = ttk.Entry(dip_frame, width=5)
        self.dip_angle_entry.insert(0, "0")
        self.dip_angle_entry.pack(side='left', padx=1)
        ttk.Label(dip_frame, text="강도:").pack(side='left', padx=(5, 1))
        self.dip_weight_entry = ttk.Entry(dip_frame, width=5)
        self.dip_weight_entry.insert(0, "1.0")
        self.dip_weight_entry.pack(side='left', padx=1)

        self.inv_auto_alpha = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="L-curve 자동 정규화",
                        variable=self.inv_auto_alpha).grid(
            row=len(params) + 6, column=0, columnspan=2, sticky='w', padx=10)

        self.inv_robust = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Robust 역산",
                        variable=self.inv_robust).grid(
            row=len(params) + 7, column=0, columnspan=2, sticky='w', padx=10)

        self.inv_grid_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="모델 블록 격자선 표시",
                        variable=self.inv_grid_mode).grid(
            row=len(params) + 8, column=0, columnspan=2, sticky='w', padx=10)

        self.inv_show_vals = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="단면도에 비저항 값 표시",
                        variable=self.inv_show_vals).grid(
            row=len(params) + 9, column=0, columnspan=2, sticky='w', padx=10)

        # 컬러 레인지 조절
        ttk.Label(ctrl, text="컬러 범위 (Ω·m):").grid(
            row=len(params) + 10, column=0, sticky='w', padx=10)
        cr_frame = ttk.Frame(ctrl)
        cr_frame.grid(row=len(params) + 10, column=1, sticky='w')
        self.cmin_entry = ttk.Entry(cr_frame, width=6)
        self.cmin_entry.insert(0, "auto")
        self.cmin_entry.pack(side='left', padx=1)
        ttk.Label(cr_frame, text="~").pack(side='left')
        self.cmax_entry = ttk.Entry(cr_frame, width=6)
        self.cmax_entry.insert(0, "auto")
        self.cmax_entry.pack(side='left', padx=1)

        # 단면도 양쪽 잘라내기
        self.clip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="양끝 전극 제거 (4개)",
                        variable=self.clip_var).grid(
            row=len(params) + 11, column=0, columnspan=2, sticky='w', padx=10)

        # 등치선 (contour)
        ttk.Label(ctrl, text="등치선 (Ω·m):").grid(
            row=len(params) + 12, column=0, sticky='w', padx=10)
        self.contour_entry = ttk.Entry(ctrl, width=15)
        self.contour_entry.grid(row=len(params) + 12, column=1, sticky='w', padx=2)
        ttk.Label(ctrl, text="  쉼표 구분 (비우면 없음)",
                  font=('맑은 고딕', 8)).grid(row=len(params) + 13, column=0,
                                               columnspan=2, sticky='w', padx=15)

        # ── 데이터 오차 모델 (학술 프레임워크) ──
        ttk.Separator(ctrl).grid(row=len(params) + 14, column=0, columnspan=2, sticky='ew', pady=3)
        ttk.Label(ctrl, text="데이터 오차 모델:",
                  font=('맑은 고딕', 9, 'bold')).grid(
            row=len(params) + 15, column=0, columnspan=2, sticky='w', padx=5)

        ttk.Label(ctrl, text="상대 오차 (%):").grid(
            row=len(params) + 16, column=0, sticky='w', padx=10)
        self.pct_error_entry = ttk.Entry(ctrl, width=8)
        self.pct_error_entry.insert(0, "5")
        self.pct_error_entry.grid(row=len(params) + 16, column=1, sticky='w', padx=5)

        ttk.Label(ctrl, text="노이즈 하한:").grid(
            row=len(params) + 17, column=0, sticky='w', padx=10)
        self.noise_floor_entry = ttk.Entry(ctrl, width=8)
        self.noise_floor_entry.insert(0, "0.001")
        self.noise_floor_entry.grid(row=len(params) + 17, column=1, sticky='w', padx=5)

        ttk.Label(ctrl, text="목표 χ²/N:").grid(
            row=len(params) + 18, column=0, sticky='w', padx=10)
        self.target_chi2_entry = ttk.Entry(ctrl, width=8)
        self.target_chi2_entry.insert(0, "1.0")
        self.target_chi2_entry.grid(row=len(params) + 18, column=1, sticky='w', padx=5)

        self.inv_occam = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Occam's cooling",
                        variable=self.inv_occam).grid(
            row=len(params) + 19, column=0, columnspan=2, sticky='w', padx=10)

        ttk.Separator(ctrl).grid(row=len(params) + 20, column=0, columnspan=2, sticky='ew', pady=3)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=len(params) + 21, column=0, columnspan=2, padx=10, pady=3, sticky='ew')
        ttk.Button(btn_frame, text="역산 실행",
                   command=self._run_inversion).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(btn_frame, text="다시 그리기",
                   command=self._redraw_inversion).pack(side='left', expand=True, fill='x', padx=2)

        save_img_frame = ttk.Frame(ctrl)
        save_img_frame.grid(row=len(params) + 22, column=0, columnspan=2, padx=10, pady=3, sticky='ew')
        ttk.Button(save_img_frame, text="전체 저장",
                   command=lambda: self._save_image('all')).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(save_img_frame, text="단면도만 저장",
                   command=lambda: self._save_image('section')).pack(side='left', expand=True, fill='x', padx=2)

        # ── 학술 분석 버튼 ──
        ttk.Separator(ctrl).grid(row=len(params) + 23, column=0, columnspan=2, sticky='ew', pady=3)
        ttk.Label(ctrl, text="학술 분석:",
                  font=('맑은 고딕', 9, 'bold')).grid(
            row=len(params) + 24, column=0, columnspan=2, sticky='w', padx=5)

        analysis_frame = ttk.Frame(ctrl)
        analysis_frame.grid(row=len(params) + 25, column=0, columnspan=2, padx=10, pady=2, sticky='ew')
        ttk.Button(analysis_frame, text="수렴 곡선",
                   command=self._show_convergence).pack(side='left', expand=True, fill='x', padx=1)
        ttk.Button(analysis_frame, text="감도 분석",
                   command=self._show_sensitivity).pack(side='left', expand=True, fill='x', padx=1)

        analysis_frame2 = ttk.Frame(ctrl)
        analysis_frame2.grid(row=len(params) + 26, column=0, columnspan=2, padx=10, pady=2, sticky='ew')
        ttk.Button(analysis_frame2, text="분해능 분석",
                   command=self._show_resolution).pack(side='left', expand=True, fill='x', padx=1)
        ttk.Button(analysis_frame2, text="결과 내보내기",
                   command=self._export_results).pack(side='left', expand=True, fill='x', padx=1)

        ttk.Button(ctrl, text="DOI 신뢰도 분석",
                   command=self._run_doi).grid(
            row=len(params) + 27, column=0, columnspan=2, pady=3, padx=10, sticky='ew')

        # 로그
        self.inv_log = tk.Text(ctrl, width=25, height=8, state='disabled',
                               font=('Courier', 9))
        self.inv_log.grid(row=len(params) + 28, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        # Plot (스크롤 가능한 큰 캔버스)
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(side='right', fill='both', expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side='top', fill='x')

        # 스크롤 영역 (가로 + 세로)
        h_scroll = ttk.Scrollbar(plot_frame, orient='horizontal')
        v_scroll = ttk.Scrollbar(plot_frame, orient='vertical')
        scroll_canvas = tk.Canvas(plot_frame,
                                  xscrollcommand=h_scroll.set,
                                  yscrollcommand=v_scroll.set)
        h_scroll.config(command=scroll_canvas.xview)
        v_scroll.config(command=scroll_canvas.yview)
        v_scroll.pack(side='right', fill='y')
        h_scroll.pack(side='bottom', fill='x')
        scroll_canvas.pack(side='left', fill='both', expand=True)

        inner_frame = ttk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=inner_frame, anchor='nw')

        # 가로: 화면 폭 - 좌측 패널(~250px) - 여백, 세로: 충분히 크게
        dpi = 100
        fig_w = max(8, (self.root.winfo_screenwidth() - 320) / dpi)
        fig_h = fig_w * 1.8  # 5패널이므로 세로 비율 크게
        self.inv_fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        canvas = FigureCanvasTkAgg(self.inv_fig, inner_frame)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame); toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.inv_canvas = canvas

        def _on_inv_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))
        inner_frame.bind('<Configure>', _on_inv_configure)

        # 마우스 휠: 세로 스크롤, Shift+휠: 가로 스크롤
        def _on_mousewheel(event):
            if event.state & 1:  # Shift 키
                scroll_canvas.xview_scroll(-1 * (event.delta // 120), 'units')
            else:
                scroll_canvas.yview_scroll(-1 * (event.delta // 120), 'units')
        scroll_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        scroll_canvas.bind_all('<Button-4>', lambda e: scroll_canvas.yview_scroll(-3, 'units'))
        scroll_canvas.bind_all('<Button-5>', lambda e: scroll_canvas.yview_scroll(3, 'units'))
        self._inv_scroll = scroll_canvas

    def _inv_log_append(self, msg):
        self.inv_log.config(state='normal')
        self.inv_log.insert('end', msg + '\n')
        self.inv_log.see('end')
        self.inv_log.config(state='disabled')
        # 팝업 로그창에도 동시 출력
        if hasattr(self, '_log_popup_text') and self._log_popup_text.winfo_exists():
            self._log_popup_text.config(state='normal')
            self._log_popup_text.insert('end', msg + '\n')
            self._log_popup_text.see('end')
            self._log_popup_text.config(state='disabled')

    def _open_log_popup(self):
        """역산 진행 상황을 보여주는 별도 팝업 창"""
        if hasattr(self, '_log_popup') and self._log_popup.winfo_exists():
            self._log_popup.lift()
            return
        popup = tk.Toplevel(self.root)
        popup.title("RESIS Pro - 역산 진행 로그")
        popup.geometry("600x500")
        popup.attributes('-topmost', True)  # 항상 위에

        txt = tk.Text(popup, font=('Courier', 10), wrap='word', state='disabled',
                      bg='#1e1e1e', fg='#d4d4d4', insertbackground='white')
        sb = ttk.Scrollbar(popup, orient='vertical', command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        txt.pack(fill='both', expand=True)

        # 기존 로그 내용 복사
        existing = self.inv_log.get('1.0', 'end').strip()
        if existing:
            txt.config(state='normal')
            txt.insert('end', existing + '\n')
            txt.see('end')
            txt.config(state='disabled')

        self._log_popup = popup
        self._log_popup_text = txt

    def _run_inversion(self):
        if self.obs_data is None or self.survey is None:
            messagebox.showerror("오류", "먼저 데이터 탭에서 APV 파일을 불러오세요.")
            return
        try:
            ref_str = self.inv_entries['inv_ref'].get().strip()
            ref = np.mean(self.obs_data) if ref_str.lower() == 'auto' else float(ref_str)
            alpha = float(self.inv_entries['inv_alpha'].get())
            ax = float(self.inv_entries['inv_ax'].get())
            az = float(self.inv_entries['inv_az'].get())
            niter = int(self.inv_entries['inv_iter'].get())
            tol = float(self.inv_entries['inv_tol'].get())
            dip_angle = float(self.dip_angle_entry.get())
            dip_weight = float(self.dip_weight_entry.get())
            pct_error = float(self.pct_error_entry.get()) / 100.0
            noise_floor = float(self.noise_floor_entry.get())
            target_chi2 = float(self.target_chi2_entry.get())
        except ValueError:
            messagebox.showerror("오류", "유효한 파라미터를 입력하세요.")
            return

        self.inv_log.config(state='normal'); self.inv_log.delete('1.0', 'end')
        self.inv_log.config(state='disabled')
        # 진행 로그 팝업 자동 열기
        self._open_log_popup()
        if hasattr(self, '_log_popup_text') and self._log_popup_text.winfo_exists():
            self._log_popup_text.config(state='normal')
            self._log_popup_text.delete('1.0', 'end')
            self._log_popup_text.config(state='disabled')
        stype = self.inv_solver_var.get()
        self._inv_log_append(f"전방 모델링: {stype}")
        self._inv_log_append(f"기준 비저항: {ref:.1f} Ω·m")
        self._inv_log_append(f"α={alpha}, αx={ax}, αz={az}")
        self._inv_log_append(f"오차 모델: {pct_error*100:.1f}% + floor={noise_floor}")
        self._inv_log_append(f"목표 χ²/N: {target_chi2:.2f}")
        self._inv_log_append(f"최대 반복: {niter}, 수렴: {tol}")
        self._inv_log_append("─" * 30)
        self.status_var.set(f"역산 실행 중 ({stype})..."); self.root.update()

        def work():
            # FEM 선택 시 삼각형 격자 생성
            tri_mesh = None
            if stype == 'FEM':
                topo = self._get_topography()
                tri_mesh = TriMesh(self.survey, topography=topo)
                self.root.after(0, lambda: self._inv_log_append(
                    f"FEM 격자: {tri_mesh.n_nodes} 노드, {tri_mesh.n_elements} 요소"))
            use_blocks = (self.inv_param_var.get() == 'block')
            reg_type = self.inv_reg_var.get()
            use_occam = self.inv_occam.get()
            inv = Inversion2D(self.survey, self.mesh, rho_ref=ref,
                              alpha=alpha, alpha_x=ax, alpha_z=az,
                              max_iter=niter, tol=tol,
                              solver_type=stype, tri_mesh=tri_mesh,
                              use_blocks=use_blocks, reg_type=reg_type,
                              dip_angle=dip_angle, dip_weight=dip_weight,
                              noise_floor=noise_floor, pct_error=pct_error,
                              target_chi2=target_chi2,
                              cooling_factor=0.7 if use_occam else 1.0)
            def cb(msg):
                self.root.after(0, lambda m=msg: [self._inv_log_append(m), self.status_var.set(m)])
            rho_inv, history, d_calc = inv.run(
                self.obs_data, callback=cb,
                auto_alpha=self.inv_auto_alpha.get(),
                robust=self.inv_robust.get())
            self._last_inv = inv  # 학술 분석용 역산 객체 보관
            self._last_inv_blocks = inv.inv_blocks
            self._last_block_rho = getattr(inv, '_last_block_rho', None)
            self.root.after(0, self._plot_inversion, rho_inv, history, d_calc)

        threading.Thread(target=work, daemon=True).start()

    def _get_color_range(self, data_fallback=None):
        """GUI에서 컬러 범위 읽기. auto는 데이터에서 자동 계산."""
        try:
            cmin_str = self.cmin_entry.get().strip().lower()
            cmax_str = self.cmax_entry.get().strip().lower()

            # 둘 다 auto면 None (완전 자동)
            if cmin_str in ('auto', '') and cmax_str in ('auto', ''):
                return None

            # 한쪽만 auto면 데이터에서 보완
            if data_fallback is not None:
                d = np.abs(data_fallback)
                auto_min, auto_max = d.min(), d.max()
            else:
                auto_min, auto_max = 1.0, 1000.0

            cmin = auto_min if cmin_str in ('auto', '') else float(cmin_str)
            cmax = auto_max if cmax_str in ('auto', '') else float(cmax_str)
            return (max(cmin, 0.1), max(cmax, cmin + 1))
        except (ValueError, AttributeError):
            return None

    def _plot_inversion(self, rho_inv, history, d_calc):
        # 결과 저장 (다시 그리기용)
        self._inv_rho = rho_inv
        self._inv_history = history
        self._inv_dcalc = d_calc
        self._redraw_inversion()
        rms = history[-1] if history else 0
        self._inv_log_append("─" * 30)
        self._inv_log_append(f"완료: {len(history)}회 반복")
        self._inv_log_append(f"최종 RMS: {rms:.4f}")
        # 학술 통계 출력
        inv = getattr(self, '_last_inv', None)
        if inv and hasattr(inv, '_final_stats'):
            st = inv._final_stats
            chi2_f = st.get('chi2_final', 0)
            alpha_f = st.get('alpha_final', 0)
            self._inv_log_append(f"최종 χ²/N: {chi2_f:.4f}")
            self._inv_log_append(f"최종 α: {alpha_f:.4e}")
            self._inv_log_append(f"N_data={st.get('n_data','?')}, "
                                 f"N_params={st.get('n_params','?')}")
            self.status_var.set(f"역산 완료: {len(history)}회, "
                                f"RMS={rms:.4f}, χ²/N={chi2_f:.3f}")
        else:
            self.status_var.set(f"역산 완료: {len(history)}회 반복, RMS = {rms:.4f}")

    def _save_image(self, mode='all'):
        """역산 결과 이미지 저장
        mode='all': 전체 3패널, 'section': 단면도만
        """
        rho_inv = getattr(self, '_inv_rho', None)
        d_calc = getattr(self, '_inv_dcalc', None)
        if rho_inv is None:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요."); return

        fp = filedialog.asksaveasfilename(
            title="이미지 저장",
            defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("TIFF", "*.tiff"), ("모든 파일", "*.*")])
        if not fp: return

        try:
            import matplotlib.pyplot as plt
            ib = getattr(self, '_last_inv_blocks', None)
            br = getattr(self, '_last_block_rho', None)

            # 컬러/등치선/클립 옵션 읽기
            all_vals = np.concatenate([np.abs(self.obs_data), np.abs(d_calc)])
            user_vr = self._get_color_range(data_fallback=all_vals)
            vr = user_vr if user_vr else (all_vals.min(), all_vals.max())
            try: clip_n = 4 if self.clip_var.get() else 0
            except: clip_n = 5
            ct_str = self.contour_entry.get().strip()
            contour_lvl = None
            if ct_str:
                try: contour_lvl = [float(v) for v in ct_str.replace(' ', ',').split(',') if v]
                except: pass

            dpi = 300  # 고해상도

            if mode == 'section':
                # 단면도만
                fig_s, ax_s = plt.subplots(figsize=(16, 6))
                plot_model_section(ax_s, self.mesh, rho_inv, self.survey,
                                   "역산 비저항 단면도",
                                   show_values=self.inv_show_vals.get(),
                                   show_block_grid=self.inv_grid_mode.get(),
                                   inv_blocks=ib, block_rho=br,
                                   global_vrange=user_vr,
                                   clip_electrodes=clip_n,
                                   contour_levels=contour_lvl)
                fig_s.tight_layout()
                fig_s.savefig(fp, dpi=dpi, bbox_inches='tight')
                plt.close(fig_s)
            else:
                # 전체 5패널
                import matplotlib.gridspec as gridspec
                fig_s = plt.figure(figsize=(16, 28))
                gs = gridspec.GridSpec(5, 2, figure=fig_s,
                                      height_ratios=[1, 1, 1, 0.8, 1.2], hspace=0.45)
                ax1 = fig_s.add_subplot(gs[0, :])
                plot_pseudosection(ax1, self.survey, self.obs_data,
                                   "관측 겉보기 비저항", global_vrange=vr)
                ax2 = fig_s.add_subplot(gs[1, :])
                plot_pseudosection(ax2, self.survey, d_calc,
                                   "계산 겉보기 비저항", global_vrange=vr)
                ax3 = fig_s.add_subplot(gs[2, :])
                plot_misfit_pseudosection(ax3, self.survey, self.obs_data, d_calc,
                                         "잔차 의사단면도")
                ax4 = fig_s.add_subplot(gs[3, :])
                plot_obs_vs_calc(ax4, self.obs_data, d_calc, "관측값 vs 계산값")
                ax5 = fig_s.add_subplot(gs[4, :])
                plot_model_section(ax5, self.mesh, rho_inv, self.survey,
                                   "역산 비저항 단면도",
                                   show_values=self.inv_show_vals.get(),
                                   show_block_grid=self.inv_grid_mode.get(),
                                   inv_blocks=ib, block_rho=br,
                                   global_vrange=user_vr,
                                   clip_electrodes=clip_n,
                                   contour_levels=contour_lvl)
                fig_s.tight_layout()
                fig_s.savefig(fp, dpi=dpi, bbox_inches='tight')
                plt.close(fig_s)

            self.status_var.set(f"이미지 저장: {fp} ({dpi}dpi)")
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{e}")

    def _redraw_inversion(self):
        """저장된 역산 결과를 현재 표시 옵션으로 다시 그리기 (역산 재실행 없음)"""
        rho_inv = getattr(self, '_inv_rho', None)
        d_calc = getattr(self, '_inv_dcalc', None)
        if rho_inv is None or d_calc is None:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요.")
            return

        try:
            fig = self.inv_fig; fig.clear()
            all_vals = np.concatenate([np.abs(self.obs_data), np.abs(d_calc)])
            user_vr = self._get_color_range(data_fallback=all_vals)
            vr = user_vr if user_vr else (all_vals.min(), all_vals.max())

            # 5패널: 관측 | 계산 | 잔차 | 산점도 | 역산 단면
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 0.8, 1.2],
                                  hspace=0.45)

            ax1 = fig.add_subplot(gs[0, :])
            plot_pseudosection(ax1, self.survey, self.obs_data, "관측 겉보기 비저항",
                               global_vrange=vr)
            ax2 = fig.add_subplot(gs[1, :])
            plot_pseudosection(ax2, self.survey, d_calc, "계산 겉보기 비저항",
                               global_vrange=vr)
            ax3 = fig.add_subplot(gs[2, :])
            plot_misfit_pseudosection(ax3, self.survey, self.obs_data, d_calc,
                                     "잔차 의사단면도")
            ax4 = fig.add_subplot(gs[3, :])
            plot_obs_vs_calc(ax4, self.obs_data, d_calc, "관측값 vs 계산값")

            ax5 = fig.add_subplot(gs[4, :])
            ib = getattr(self, '_last_inv_blocks', None)
            br = getattr(self, '_last_block_rho', None)
            try:
                clip_n = 4 if self.clip_var.get() else 0
            except ValueError:
                clip_n = 5
            contour_lvl = None
            ct_str = self.contour_entry.get().strip()
            if ct_str:
                try:
                    contour_lvl = [float(v) for v in ct_str.replace(' ', ',').split(',') if v]
                except ValueError:
                    contour_lvl = None

            plot_model_section(ax5, self.mesh, rho_inv, self.survey, "역산 비저항 단면도",
                               show_values=self.inv_show_vals.get(),
                               show_block_grid=self.inv_grid_mode.get(),
                               inv_blocks=ib, block_rho=br,
                               global_vrange=user_vr,
                               clip_electrodes=clip_n,
                               contour_levels=contour_lvl)
            fig.tight_layout(); self.inv_canvas.draw()
            self.status_var.set("그래프 갱신 완료")
        except Exception as e:
            self.status_var.set(f"그리기 오류: {e}")
            messagebox.showerror("오류", f"그래프 갱신 실패:\n{e}")

    def _geo_dip_direction_sign(self, result):
        """n-level 중심 이동으로 경사 방향 부호 추정. 불확실하면 +1."""
        try:
            cents = result.get('diag', {}).get('M2', {}).get('centroids', [])
            if len(cents) >= 3:
                xs = np.array([float(c[1]) for c in cents])
                zs = np.array([float(c[2]) for c in cents])
                slope, _ = np.polyfit(xs, zs, 1)
                if np.isfinite(slope) and abs(slope) > 1e-6:
                    return 1.0 if slope > 0 else -1.0
        except Exception:
            pass
        return 1.0

    def _draw_geo_dip_line(self, ax, angle_deg, sign, color, label, linestyle='-',
                           linewidth=2.4, alpha=0.95):
        if angle_deg is None or not np.isfinite(angle_deg) or abs(angle_deg) < 1:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        z_top = min(y0, y1)
        z_bot = max(y0, y1)
        x_mid = 0.5 * (x0 + x1)
        z_mid = z_top + 0.38 * (z_bot - z_top)
        length = 0.62 * (x1 - x0)
        half = 0.5 * length
        theta = np.radians(abs(angle_deg))
        xs = np.array([x_mid - half, x_mid + half])
        zs = z_mid + sign * np.tan(theta) * (xs - x_mid)

        # 보이는 영역 안으로 대략 제한
        valid = (zs >= z_top) & (zs <= z_bot)
        if valid.sum() < 2:
            # 너무 급하거나 중앙이 맞지 않으면 선 길이를 줄인다.
            half = 0.25 * (x1 - x0)
            xs = np.array([x_mid - half, x_mid + half])
            zs = z_mid + sign * np.tan(theta) * (xs - x_mid)

        ax.plot(xs, zs, color=color, linestyle=linestyle, lw=linewidth,
                alpha=alpha, zorder=30, solid_capstyle='round')
        ax.text(xs[-1], zs[-1], f" {label} {angle_deg:.1f}°",
                color=color, fontsize=8, fontweight='bold',
                va='center', ha='left',
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.78,
                          boxstyle='round,pad=0.18'),
                zorder=31)

    def _overlay_geological_interpretation(self, ax):
        """역산 단면도 위에 지질구조 해석 결과를 오버레이."""
        if not getattr(self, 'show_geo_overlay', tk.BooleanVar(value=False)).get():
            return
        result = getattr(self, '_geo_interp_result', None)
        if not result:
            return
        try:
            pred = result.get('prediction', {})
            rec = result.get('recommendation', {})
            hm = result.get('hypothesis_match') or {}
            sign = self._geo_dip_direction_sign(result)

            final_angle = float(pred.get('estimate', np.nan))
            self._draw_geo_dip_line(
                ax, final_angle, sign, color='#ffd21f',
                label='통합해석', linestyle='-', linewidth=2.8, alpha=0.98)

            if hm and not hm.get('error'):
                hyp_angle = hm.get('dip_estimate')
                hyp_corr = float(hm.get('best_corr', 0.0))
                if hyp_angle is not None and hyp_corr >= 0.60:
                    self._draw_geo_dip_line(
                        ax, float(hyp_angle), +1.0, color='#00bcd4',
                        label='Forward후보', linestyle=(0, (7, 4)),
                        linewidth=2.2, alpha=0.88)

            # 요약 박스
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            z_top = min(y0, y1)
            z_bot = max(y0, y1)
            hm_txt = '없음'
            if hm and not hm.get('error'):
                hd = hm.get('dip_estimate')
                hd_txt = '단일각 없음' if hd is None else f"{hd:.1f}°"
                hm_txt = f"{hm.get('best_family','')} / {hd_txt} / r={hm.get('best_corr',0):.2f}"

            txt = (
                "지질구조 해석\n"
                f"통합: {pred.get('estimate', 0):.1f}±{pred.get('uncertainty', 0):.1f}°\n"
                f"방법: {pred.get('method','')}\n"
                f"Forward: {hm_txt}\n"
                f"추천: {rec.get('mode','')}"
            )
            ax.text(x0 + 0.02 * (x1 - x0), z_top + 0.08 * (z_bot - z_top),
                    txt, fontsize=8.2, color='black', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='#333333',
                              alpha=0.82, boxstyle='round,pad=0.35'),
                    zorder=32)
        except Exception as e:
            self._inv_log_append(f"지질구조 오버레이 생략: {e}")

    def _run_doi(self):
        if self.obs_data is None or self.survey is None or self.mesh is None:
            messagebox.showerror("오류", "먼저 데이터를 불러오고 역산을 실행하세요.")
            return
        self.inv_log.config(state='normal'); self.inv_log.delete('1.0', 'end')
        self.inv_log.config(state='disabled')
        self.status_var.set("DOI 분석 중 (2회 역산)..."); self.root.update()

        def work():
            stype = self.inv_solver_var.get()
            tri_mesh = None
            if stype == 'FEM':
                tri_mesh = TriMesh(self.survey, topography=self._get_topography())
            def cb(msg):
                self.root.after(0, lambda m=msg: [self._inv_log_append(m), self.status_var.set(m)])
            doi = compute_doi(self.survey, self.mesh, self.obs_data,
                              rho_ref=np.mean(self.obs_data), max_iter=4, callback=cb,
                              auto_alpha=self.inv_auto_alpha.get(),
                              robust=self.inv_robust.get(),
                              solver_type=stype, tri_mesh=tri_mesh)
            self.root.after(0, self._plot_doi, doi)
        threading.Thread(target=work, daemon=True).start()

    def _plot_doi(self, doi):
        fig = self.inv_fig; fig.clear()
        m = self.mesh; a = self.survey.a; ex = self.survey.electrode_x
        ncx, ncz = m.ncx, m.ncz
        doi_2d = doi.reshape(ncz, ncx)

        x_lim = (ex[0] - a, ex[-1] + a)
        z_lim = self.survey.n_max * a * 1.3
        mask_x = (m.x_nodes >= x_lim[0]) & (m.x_nodes <= x_lim[1])
        mask_z = m.z_nodes <= z_lim
        ix0 = np.where(mask_x)[0][0]; ix1 = np.where(mask_x)[0][-1]
        iz1 = np.where(mask_z)[0][-1]
        cx = m.x_cc[ix0:ix1]; cz = m.z_cc[:iz1]
        doi_sub = doi_2d[:iz1, ix0:ix1]

        xi = np.linspace(cx[0], cx[-1], 400)
        zi = np.linspace(cz[0], cz[-1], 200)
        XI, ZI = np.meshgrid(xi, zi)
        CX, CZ = np.meshgrid(cx, cz)
        VI = griddata((CX.ravel(), CZ.ravel()), doi_sub.ravel(), (XI, ZI), method='cubic')
        VI_fill = griddata((CX.ravel(), CZ.ravel()), doi_sub.ravel(), (XI, ZI), method='nearest')
        mask = np.isnan(VI); VI[mask] = VI_fill[mask]
        # 사다리꼴 마스크
        z_max = self.survey.n_max * a * 1.25
        left_s = ex[0] + ZI * 0.7; right_s = ex[-1] - ZI * 0.7
        outside = (XI < left_s) | (XI > right_s) | (ZI > z_max)
        VI[outside] = np.nan

        ax = fig.add_subplot(111)
        im = ax.pcolormesh(XI, ZI, VI, cmap='RdYlGn_r', vmin=0, vmax=0.5,
                           shading='gouraud', rasterized=True)
        # 등치선: DOI = 0.1, 0.2, 0.3
        cs = ax.contour(XI, ZI, VI, levels=[0.1, 0.2, 0.3],
                        colors=['green', 'orange', 'red'], linewidths=1.5)
        ax.clabel(cs, fmt='%.1f', fontsize=8)
        for i, xe in enumerate(ex):
            ax.plot(xe, 0, 'kv', markersize=4, zorder=10)
        ax.set_xlim(ex[0] - a, ex[-1] + a)
        ax.set_ylim(z_max + a * 0.5, -a * 0.6)
        ax.set_xlabel('거리 (m)'); ax.set_ylabel('깊이 (m)')
        ax.set_title('DOI 지수 (Depth of Investigation)\n'
                     'DOI < 0.1: 신뢰 / 0.1~0.3: 보통 / > 0.3: 불확실',
                     fontweight='bold', fontsize=11)
        ax.set_aspect('equal')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.45)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label('DOI 지수 (0=신뢰, 1=불확실)')
        fig.tight_layout(); self.inv_canvas.draw()
        self.status_var.set("DOI 분석 완료")

    # ── 학술 분석 메서드 ──

    def _run_geological_interpretation(self):
        """현재 자료로 지질구조 후보 해석 실행."""
        if self.obs_data is None or self.survey is None:
            messagebox.showerror("오류", "먼저 데이터를 불러오세요.")
            return
        if self.mesh is None:
            self.mesh = self._make_mesh(self.survey)

        self._open_log_popup()
        self._inv_log_append("─" * 30)
        self._inv_log_append("지질구조 해석 시작")
        self._inv_log_append("진단 + ML/OOD + forward hypothesis matching")
        self.status_var.set("지질구조 해석 중..."); self.root.update()

        def work():
            try:
                from pathlib import Path
                from geo_structure_interpreter import (
                    ROOT, interpret_survey_data, load_ml_model)

                outdir = ROOT / "GeoInterp_outputs"
                model_dict = load_ml_model()
                name = "RESIS_current"
                if self.apv_info is not None:
                    area = str(self.apv_info.get('area', '')).strip()
                    line = str(self.apv_info.get('line', '')).strip()
                    name = "_".join([v for v in [area, line] if v]) or name

                result = interpret_survey_data(
                    self.survey, self.obs_data, outdir,
                    name=name, model_dict=model_dict, mesh=self.mesh,
                    use_hypothesis=True, source_path=name)

                self.root.after(0, self._finish_geological_interpretation, result, outdir)
            except Exception as e:
                self.root.after(0, lambda err=e: [
                    self._inv_log_append(f"지질구조 해석 오류: {err}"),
                    self.status_var.set("지질구조 해석 실패"),
                    messagebox.showerror("오류", f"지질구조 해석 실패:\n{err}")
                ])

        threading.Thread(target=work, daemon=True).start()

    def _finish_geological_interpretation(self, result, outdir):
        self._geo_interp_result = result
        pred = result['prediction']
        rec = result['recommendation']
        hm = result.get('hypothesis_match') or {}

        self._inv_log_append("지질구조 해석 완료")
        self._inv_log_append(f"진단 추정: {pred['estimate']:.1f} ± {pred['uncertainty']:.1f}°")
        self._inv_log_append(f"방법: {pred['method']}, 신뢰도={pred['confidence']:.2f}")
        if hm and not hm.get('error'):
            hd = hm.get('dip_estimate')
            hd_txt = "단일각 없음" if hd is None else f"{hd:.1f}°"
            self._inv_log_append(
                f"Forward 후보: {hm.get('best_family','')} / {hd_txt} "
                f"(corr={hm.get('best_corr',0):.2f})")
        elif hm and hm.get('error'):
            self._inv_log_append(f"Forward 후보 매칭 실패: {hm['error']}")
        self._inv_log_append(f"추천 모드: {rec['mode']}")
        self._inv_log_append(f"결과 폴더: {outdir}")

        # 경사 평활 입력값에는 단일 증거가 아니라 최종 통합 추정값을 반영한다.
        # Forward 후보각은 로그/리포트에 남기고, 자동 preset은 보수적으로 둔다.
        try:
            apply_angle = pred['estimate']
            if apply_angle is not None and np.isfinite(apply_angle):
                self.dip_angle_entry.delete(0, 'end')
                self.dip_angle_entry.insert(0, f"{float(apply_angle):.1f}")
                self.dip_weight_entry.delete(0, 'end')
                self.dip_weight_entry.insert(0, "3.0")
        except Exception:
            pass

        msg = (
            f"지질구조 해석 완료\n\n"
            f"진단 추정: {pred['estimate']:.1f} ± {pred['uncertainty']:.1f}°\n"
            f"방법: {pred['method']}\n"
        )
        if hm and not hm.get('error'):
            hd = hm.get('dip_estimate')
            hd_txt = "단일각 없음" if hd is None else f"{hd:.1f}°"
            msg += (
                f"Forward 후보: {hm.get('best_family','')}\n"
                f"Forward 경사: {hd_txt}, corr={hm.get('best_corr',0):.2f}\n"
            )
        msg += (
            f"추천 모드: {rec['mode']}\n"
            f"후보 범위: {rec['dip_range'][0]:.1f}° - {rec['dip_range'][1]:.1f}°\n\n"
            f"결과 저장:\n{outdir}"
        )
        self.status_var.set("지질구조 해석 완료")
        if getattr(self, '_inv_rho', None) is not None and getattr(self, '_inv_dcalc', None) is not None:
            try:
                self._redraw_inversion()
            except Exception:
                pass
        messagebox.showinfo("지질구조 해석", msg)

    def _show_convergence(self):
        """수렴 진단 4패널 표시"""
        inv = getattr(self, '_last_inv', None)
        if inv is None or not inv.convergence['rms']:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요.")
            return
        fig = self.inv_fig; fig.clear()
        target = float(self.target_chi2_entry.get()) if hasattr(self, 'target_chi2_entry') else 1.0
        plot_convergence(fig, inv.convergence, target_chi2=target)
        fig.tight_layout()
        self.inv_canvas.draw()
        self.status_var.set("수렴 진단 곡선 표시")

    def _show_sensitivity(self):
        """누적 감도 단면도 표시"""
        inv = getattr(self, '_last_inv', None)
        if inv is None:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요.")
            return
        try:
            sens = inv.compute_sensitivity()
        except RuntimeError as e:
            messagebox.showerror("오류", str(e)); return

        fig = self.inv_fig; fig.clear()
        ax = fig.add_subplot(111)
        ib = getattr(self, '_last_inv_blocks', None)
        try: clip_n = 4 if self.clip_var.get() else 0
        except: clip_n = 5
        plot_sensitivity_section(ax, self.mesh, sens, self.survey,
                                 title="Cumulative Sensitivity (Friedel, 2003)",
                                 inv_blocks=ib, clip_electrodes=clip_n)
        fig.tight_layout()
        self.inv_canvas.draw()
        self.status_var.set("감도 분석 완료")

    def _show_resolution(self):
        """모델 분해능 대각 성분 단면도 표시"""
        inv = getattr(self, '_last_inv', None)
        if inv is None:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요.")
            return
        self.status_var.set("분해능 행렬 계산 중..."); self.root.update()
        try:
            res_diag = inv.compute_resolution_diagonal()
        except RuntimeError as e:
            messagebox.showerror("오류", str(e)); return

        fig = self.inv_fig; fig.clear()
        ax = fig.add_subplot(111)
        ib = getattr(self, '_last_inv_blocks', None)
        try: clip_n = 4 if self.clip_var.get() else 0
        except: clip_n = 5
        plot_resolution_section(ax, self.mesh, res_diag, self.survey,
                                title="Model Resolution (Menke, 2012)",
                                inv_blocks=ib, clip_electrodes=clip_n)
        fig.tight_layout()
        self.inv_canvas.draw()
        self.status_var.set(f"분해능 분석 완료: 평균 R = {np.mean(res_diag):.3f}")

    def _export_results(self):
        """역산 결과를 XYZ + CSV로 내보내기"""
        inv = getattr(self, '_last_inv', None)
        rho_inv = getattr(self, '_inv_rho', None)
        d_calc = getattr(self, '_inv_dcalc', None)
        if inv is None or rho_inv is None:
            messagebox.showinfo("알림", "먼저 역산을 실행하세요.")
            return
        fp = filedialog.asksaveasfilename(
            title="결과 내보내기 (기본 파일명)",
            defaultextension='.xyz',
            filetypes=[("XYZ 파일", "*.xyz"), ("모든 파일", "*.*")])
        if not fp: return
        try:
            base = inv.export_results(fp, rho_inv, self.obs_data, d_calc)
            self.status_var.set(f"내보내기 완료: {base}_model.xyz, _data.xyz, _convergence.csv")
            messagebox.showinfo("완료",
                f"3개 파일 생성:\n"
                f"  {os.path.basename(base)}_model.xyz\n"
                f"  {os.path.basename(base)}_data.xyz\n"
                f"  {os.path.basename(base)}_convergence.csv")
        except Exception as e:
            messagebox.showerror("오류", f"내보내기 실패:\n{e}")

    # ── Tab 3: 전방 모델링 ──
    def _create_forward_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  전방 모델링  ")
        ctrl = ttk.LabelFrame(tab, text="모델 설정")
        ctrl.pack(side='left', fill='y', padx=5, pady=5)

        ttk.Label(ctrl, text="전극 간격 a (m):").grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.fwd_a = ttk.Entry(ctrl, width=10); self.fwd_a.insert(0, "5"); self.fwd_a.grid(row=0, column=1)
        ttk.Label(ctrl, text="전극 수:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.fwd_ne = ttk.Entry(ctrl, width=10); self.fwd_ne.insert(0, "29"); self.fwd_ne.grid(row=1, column=1)
        ttk.Label(ctrl, text="최대 n-level:").grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.fwd_nm = ttk.Entry(ctrl, width=10); self.fwd_nm.insert(0, "4"); self.fwd_nm.grid(row=2, column=1)
        ttk.Label(ctrl, text="배경 비저항 (Ω·m):").grid(row=3, column=0, sticky='w', padx=5, pady=3)
        self.fwd_bg = ttk.Entry(ctrl, width=10); self.fwd_bg.insert(0, "100"); self.fwd_bg.grid(row=3, column=1)

        ttk.Label(ctrl, text="이상체 (x1 z1 x2 z2 ρ):").grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        cols = ("x1", "z1", "x2", "z2", "ρ")
        self.fwd_tree = ttk.Treeview(ctrl, columns=cols, show='headings', height=6)
        for c in cols:
            self.fwd_tree.heading(c, text=c); self.fwd_tree.column(c, width=50, anchor='center')
        self.fwd_tree.grid(row=5, column=0, columnspan=2, padx=5)

        ef = ttk.Frame(ctrl); ef.grid(row=6, column=0, columnspan=2, pady=3)
        self.fwd_block_entries = []
        for c in cols:
            e = ttk.Entry(ef, width=6); e.pack(side='left', padx=1)
            self.fwd_block_entries.append(e)
        bf = ttk.Frame(ctrl); bf.grid(row=7, column=0, columnspan=2, pady=3)
        ttk.Button(bf, text="추가", command=self._add_block).pack(side='left', padx=3)
        ttk.Button(bf, text="삭제", command=self._del_block).pack(side='left', padx=3)

        ttk.Separator(ctrl).grid(row=8, column=0, columnspan=2, sticky='ew', pady=5)

        # 마우스 그리기 모드
        self.fwd_draw_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="마우스로 모델 그리기",
                        variable=self.fwd_draw_mode,
                        command=self._toggle_draw_mode).grid(
            row=9, column=0, columnspan=2, sticky='w', padx=5)
        ttk.Label(ctrl, text="  드래그: 영역 선택 → 비저항 입력",
                  font=('맑은 고딕', 8)).grid(row=10, column=0, columnspan=2, sticky='w', padx=10)

        ttk.Button(ctrl, text="모델 미리보기",
                   command=self._preview_model).grid(row=11, column=0, columnspan=2, pady=5)
        ttk.Button(ctrl, text="전방 모델링 실행",
                   command=self._run_forward).grid(row=12, column=0, columnspan=2, pady=5)
        ttk.Button(ctrl, text="→ 역산 데이터로 전송 (3% 노이즈)",
                   command=self._send_to_inv).grid(row=13, column=0, columnspan=2, pady=5)

        self.fwd_fig = Figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(self.fwd_fig, tab)
        toolbar = NavigationToolbar2Tk(canvas, tab); toolbar.update()
        canvas.get_tk_widget().pack(side='right', fill='both', expand=True)
        self.fwd_canvas = canvas
        self.fwd_rho_a = None

        # 마우스 이벤트 상태
        self._drag_start = None
        self._drag_rect = None
        self._fwd_cids = []

    def _add_block(self):
        vals = []
        for e in self.fwd_block_entries:
            try: vals.append(float(e.get()))
            except: messagebox.showerror("오류", "숫자를 입력하세요."); return
        self.fwd_tree.insert('', 'end', values=tuple(vals))
        for e in self.fwd_block_entries: e.delete(0, 'end')

    def _del_block(self):
        for s in self.fwd_tree.selection(): self.fwd_tree.delete(s)

    def _toggle_draw_mode(self):
        """마우스 그리기 모드 토글"""
        if self.fwd_draw_mode.get():
            # 먼저 모델 미리보기 표시
            self._preview_model()
            # 마우스 이벤트 연결
            fig = self.fwd_fig
            cid1 = fig.canvas.mpl_connect('button_press_event', self._on_fwd_press)
            cid2 = fig.canvas.mpl_connect('button_release_event', self._on_fwd_release)
            cid3 = fig.canvas.mpl_connect('motion_notify_event', self._on_fwd_motion)
            self._fwd_cids = [cid1, cid2, cid3]
            self.status_var.set("그리기 모드: 단면도에서 드래그하여 이상체 추가")
        else:
            # 마우스 이벤트 해제
            for cid in self._fwd_cids:
                self.fwd_fig.canvas.mpl_disconnect(cid)
            self._fwd_cids = []
            self._drag_rect = None
            self.status_var.set("그리기 모드 해제")

    def _on_fwd_press(self, event):
        """마우스 클릭: 드래그 시작점 기록"""
        if event.inaxes is None or event.button != 1:
            return
        self._drag_start = (event.xdata, event.ydata)
        # 기존 rubber band 제거
        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None

    def _on_fwd_motion(self, event):
        """마우스 이동: rubber band 직사각형 표시"""
        if self._drag_start is None or event.inaxes is None:
            return
        x0, z0 = self._drag_start
        x1, z1 = event.xdata, event.ydata
        # 기존 사각형 제거
        if self._drag_rect is not None:
            self._drag_rect.remove()
        self._drag_rect = Rectangle(
            (min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, zorder=20)
        event.inaxes.add_patch(self._drag_rect)
        self.fwd_canvas.draw_idle()

    def _on_fwd_release(self, event):
        """마우스 릴리스: 영역 확정 → 비저항 입력"""
        if self._drag_start is None or event.inaxes is None:
            return
        x0, z0 = self._drag_start
        x1, z1 = event.xdata, event.ydata
        self._drag_start = None

        # 너무 작은 영역 무시
        if abs(x1 - x0) < 0.5 or abs(z1 - z0) < 0.3:
            if self._drag_rect:
                self._drag_rect.remove(); self._drag_rect = None
                self.fwd_canvas.draw_idle()
            return

        # 비저항 입력 다이얼로그
        from tkinter import simpledialog
        rho_val = simpledialog.askfloat("비저항 입력",
            f"영역 ({min(x0,x1):.1f},{min(z0,z1):.1f})~({max(x0,x1):.1f},{max(z0,z1):.1f})\n"
            f"비저항 (Ω·m):", parent=self.root)

        # rubber band 제거
        if self._drag_rect:
            self._drag_rect.remove(); self._drag_rect = None

        if rho_val is not None and rho_val > 0:
            # fwd_tree에 추가
            vals = (f"{min(x0,x1):.1f}", f"{min(z0,z1):.1f}",
                    f"{max(x0,x1):.1f}", f"{max(z0,z1):.1f}", f"{rho_val:.1f}")
            self.fwd_tree.insert('', 'end', values=vals)
            # 모델 미리보기 갱신
            self._preview_model()
        else:
            self.fwd_canvas.draw_idle()

    def _preview_model(self):
        """현재 설정으로 모델 단면도 미리보기 (전방 모델링 없이)"""
        try:
            a = float(self.fwd_a.get()); ne = int(self.fwd_ne.get())
            nm = int(self.fwd_nm.get()); bg = float(self.fwd_bg.get())
        except ValueError:
            messagebox.showerror("오류", "파라미터를 확인하세요."); return

        sv = DipDipSurvey(a=a, n_electrodes=ne, n_max=nm)
        mesh = Mesh2D(sv, dx_factor=0.5)  # 미리보기는 빠른 격자
        rho = np.full(mesh.n_cells, bg)

        for item in self.fwd_tree.get_children():
            v = [float(x) for x in self.fwd_tree.item(item, 'values')]
            x1, z1, x2, z2, r = v
            for iz in range(mesh.ncz):
                for ix in range(mesh.ncx):
                    if x1 <= mesh.x_cc[ix] <= x2 and z1 <= mesh.z_cc[iz] <= z2:
                        rho[mesh.cidx(ix, iz)] = r

        fig = self.fwd_fig; fig.clear()
        ax = fig.add_subplot(111)
        # 셀 직접 표시 (보간 없이 빠르게)
        ncx, ncz = mesh.ncx, mesh.ncz
        rho_2d = rho.reshape(ncz, ncx)
        ex = sv.electrode_x
        z_max = nm * a * 1.25
        X, Z = np.meshgrid(mesh.x_nodes, mesh.z_nodes)
        norm = LogNorm(vmin=max(rho.min(), 0.1), vmax=rho.max())
        ax.pcolormesh(X, Z, rho_2d, cmap=RHO_CMAP, norm=norm, shading='flat')
        # 전극 표시
        for i, xe in enumerate(ex):
            ax.plot(xe, 0, 'kv', ms=3, zorder=10)
        ax.set_xlim(ex[0] - a, ex[-1] + a)
        ax.set_ylim(z_max + a * 0.3, -a * 0.5)
        ax.set_xlabel('거리 (m)'); ax.set_ylabel('깊이 (m)')
        ax.set_title('모델 미리보기 (드래그로 이상체 추가)', fontweight='bold')
        ax.set_aspect('equal')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes('bottom', size='5%', pad=0.45)
        fig.colorbar(plt.cm.ScalarMappable(cmap=RHO_CMAP, norm=norm),
                     cax=cax, orientation='horizontal', label='비저항 (Ω·m)')
        fig.tight_layout()
        self.fwd_canvas.draw()

    def _run_forward(self):
        try:
            a = float(self.fwd_a.get()); ne = int(self.fwd_ne.get())
            nm = int(self.fwd_nm.get()); bg = float(self.fwd_bg.get())
        except: messagebox.showerror("오류", "파라미터를 확인하세요."); return

        self.survey = DipDipSurvey(a=a, n_electrodes=ne, n_max=nm)
        self.mesh = self._make_mesh(self.survey)
        rho = np.full(self.mesh.n_cells, bg)
        for item in self.fwd_tree.get_children():
            v = [float(x) for x in self.fwd_tree.item(item, 'values')]
            x1, z1, x2, z2, r = v
            for iz in range(self.mesh.ncz):
                for ix in range(self.mesh.ncx):
                    if x1 <= self.mesh.x_cc[ix] <= x2 and z1 <= self.mesh.z_cc[iz] <= z2:
                        rho[self.mesh.cidx(ix, iz)] = r
        self.status_var.set("전방 모델링 중..."); self.root.update()
        self._fwd_model = rho

        def work():
            solver = ForwardSolver(self.mesh, rho)
            self.fwd_rho_a = solver.compute_data(self.survey)
            self.root.after(0, self._plot_forward)
        threading.Thread(target=work, daemon=True).start()

    def _plot_forward(self):
        fig = self.fwd_fig; fig.clear()
        ax1 = fig.add_subplot(211)
        plot_model_section(ax1, self.mesh, self._fwd_model, self.survey, "입력 비저항 모델")
        ax2 = fig.add_subplot(212)
        plot_pseudosection(ax2, self.survey, self.fwd_rho_a, "겉보기 비저항 의사단면도")
        fig.tight_layout(); self.fwd_canvas.draw()
        self.status_var.set(f"전방 모델링 완료 ({self.survey.n_data}개 측정)")

    def _send_to_inv(self):
        if self.fwd_rho_a is None:
            messagebox.showinfo("알림", "전방 모델링을 먼저 실행하세요."); return
        self.obs_data = self.fwd_rho_a + 0.03 * self.fwd_rho_a * np.random.randn(len(self.fwd_rho_a))
        self.notebook.select(1)
        self.status_var.set("전방 모델링 데이터 (3% 노이즈)가 역산 탭으로 전송됨")


# ============================================================
def main():
    root = tk.Tk()
    app = DipoleDipoleApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

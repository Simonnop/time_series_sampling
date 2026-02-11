#!/usr/bin/env python3
"""
合并的绘图脚本，包含生成卷积动画图和生成 Makefile 的功能。
"""
import argparse
import itertools
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _get_pdflatex_cmd() -> Optional[List[str]]:
    """返回 pdflatex 命令列表，未找到时返回 None。"""
    exe = shutil.which('pdflatex')
    return [exe] if exe else None


def _get_convert_cmd() -> Optional[List[str]]:
    """返回 ImageMagick convert 命令列表（兼容 6 与 7），未找到时返回 None。"""
    exe = shutil.which('convert')
    if exe:
        return [exe]
    magick = shutil.which('magick')
    if magick:
        return [magick, 'convert']
    return None

np.random.seed(1234)


def make_numerical_tex_string(
    step: int,
    input_size: int,
    output_size: int,
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    mode: str,
) -> bytes:
    """创建数值卷积动画的 LaTeX 字符串。

    参数
    ----------
    step : int
        动画的步骤编号。
    input_size : int
        卷积输入大小。
    output_size : int
        卷积输出大小。
    padding : int
        零填充。
    kernel_size : int
        卷积核大小。
    stride : int
        卷积步长。
    dilation : int
        输入膨胀。
    mode : str
        核模式，可选 {'convolution', 'average', 'max'}。

    返回
    -------
    tex_string : bytes
        可被 LaTeX 编译的字符串，用于生成动画的一步。
    """
    if mode not in ('convolution', 'average', 'max'):
        raise ValueError(
            "错误的卷积模式，可选 'convolution', 'average' 或 'max'"
        )
    if dilation != 1:
        raise ValueError("目前仅支持膨胀为 1 的数值输出")
    
    max_steps = output_size ** 2
    if step >= max_steps:
        raise ValueError(
            f'步骤 {step} 超出范围（此动画共有 {max_steps} 步）'
        )

    template_path = Path('templates') / 'numerical_figure.txt'
    with open(template_path, 'r', encoding='utf-8') as f:
        tex_template = f.read()

    total_input_size = input_size + 2 * padding

    input_ = np.zeros((total_input_size, total_input_size), dtype='int32')
    input_[padding: padding + input_size,
           padding: padding + input_size] = np.random.randint(
        low=0, high=4, size=(input_size, input_size))
    kernel = np.random.randint(
        low=0, high=3, size=(kernel_size, kernel_size))
    output = np.empty((output_size, output_size), dtype='float32')
    
    for offset_x, offset_y in itertools.product(range(output_size),
                                                range(output_size)):
        if mode == 'convolution':
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size] * kernel).sum()
        elif mode == 'average':
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size]).mean()
        else:  # max
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size]).max()

    offsets = list(itertools.product(range(output_size - 1, -1, -1),
                                     range(output_size)))
    offset_y, offset_x = offsets[step]

    if mode == 'convolution':
        kernel_values_string = ''.join(
            f"\\node (node) at ({i + 0.8 + stride * offset_x},{j + 0.2 + stride * offset_y}) {{\\tiny {kernel[kernel_size - 1 - j, i]}}};\n"
            for i, j in itertools.product(range(kernel_size),
                                          range(kernel_size)))
    else:
        kernel_values_string = '\n'

    return tex_template.format(
        PADDING_TO=f'{total_input_size},{total_input_size}',
        INPUT_FROM=f'{padding},{padding}',
        INPUT_TO=f'{padding + input_size},{padding + input_size}',
        INPUT_VALUES=''.join(
            f"\\node (node) at ({i + 0.5},{j + 0.5}) {{\\footnotesize {input_[total_input_size - 1 - j, i]}}};\n"
            for i, j in itertools.product(range(total_input_size),
                                          range(total_input_size))),
        INPUT_GRID_FROM=f'{stride * offset_x},{stride * offset_y}',
        INPUT_GRID_TO=f'{stride * offset_x + kernel_size},{stride * offset_y + kernel_size}',
        KERNEL_VALUES=kernel_values_string,
        OUTPUT_TO=f'{output_size},{output_size}',
        OUTPUT_GRID_FROM=f'{offset_x},{offset_y}',
        OUTPUT_GRID_TO=f'{offset_x + 1},{offset_y + 1}',
        OUTPUT_VALUES=''.join(
            f"\\node (node) at ({i + 0.5},{j + 0.5}) {{\\tiny {output[output_size - 1 - j, i]:.1f}}};\n"
            for i, j in itertools.product(range(output_size),
                                          range(output_size))),
        XSHIFT=f'{total_input_size + 1}cm',
        YSHIFT=f'{(total_input_size - output_size) // 2}cm',
    ).encode('utf-8')


def make_arithmetic_tex_string(
    step: int,
    input_size: int,
    output_size: int,
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    transposed: bool,
) -> bytes:
    """创建卷积算术动画的 LaTeX 字符串。

    参数
    ----------
    step : int
        动画的步骤编号。
    input_size : int
        卷积输入大小。
    output_size : int
        卷积输出大小。
    padding : int
        零填充。
    kernel_size : int
        卷积核大小。
    stride : int
        卷积步长。
    dilation : int
        输入膨胀。
    transposed : bool
        如果为 ``True``，生成转置卷积动画的字符串。

    返回
    -------
    tex_string : bytes
        可被 LaTeX 编译的字符串，用于生成动画的一步。
    """
    kernel_size = (kernel_size - 1) * dilation + 1
    if transposed:
        # 用于添加底部填充以处理奇数形状
        bottom_pad = (input_size + 2 * padding - kernel_size) % stride

        input_size, output_size, padding, spacing, stride = (
            output_size, input_size, kernel_size - 1 - padding, stride, 1)
        total_input_size = output_size + kernel_size - 1
        y_adjustment = 0
    else:
        # 在卷积中不使用
        bottom_pad = 0

        spacing = 1
        total_input_size = input_size + 2 * padding
        y_adjustment = (total_input_size - (kernel_size - stride)) % stride

    max_steps = output_size ** 2
    if step >= max_steps:
        raise ValueError(
            f'步骤 {step} 超出范围（此动画共有 {max_steps} 步）'
        )

    arithmetic_template_path = Path('templates') / 'arithmetic_figure.txt'
    unit_template_path = Path('templates') / 'unit.txt'
    
    with open(arithmetic_template_path, 'r', encoding='utf-8') as f:
        tex_template = f.read()
    with open(unit_template_path, 'r', encoding='utf-8') as f:
        unit_template = f.read()

    offsets = list(itertools.product(range(output_size - 1, -1, -1),
                                     range(output_size)))
    offset_y, offset_x = offsets[step]

    return tex_template.format(
        PADDING_TO=f'{total_input_size},{total_input_size}',
        INPUT_UNITS=''.join(
            unit_template.format(
                padding + spacing * i,
                bottom_pad + padding + spacing * j,
                padding + spacing * i + 1,
                bottom_pad + padding + spacing * j + 1)
            for i, j in itertools.product(range(input_size),
                                          range(input_size))),
        INPUT_GRID_FROM_X=f'{stride * offset_x}',
        INPUT_GRID_FROM_Y=f'{y_adjustment + stride * offset_y}',
        INPUT_GRID_TO_X=f'{stride * offset_x + kernel_size}',
        INPUT_GRID_TO_Y=f'{y_adjustment + stride * offset_y + kernel_size}',
        DILATION=f'{dilation}',
        OUTPUT_BOTTOM_LEFT=f'{offset_x},{offset_y}',
        OUTPUT_BOTTOM_RIGHT=f'{offset_x + 1},{offset_y}',
        OUTPUT_TOP_LEFT=f'{offset_x},{offset_y + 1}',
        OUTPUT_TOP_RIGHT=f'{offset_x + 1},{offset_y + 1}',
        OUTPUT_TO=f'{output_size},{output_size}',
        OUTPUT_GRID_FROM=f'{offset_x},{offset_y}',
        OUTPUT_GRID_TO=f'{offset_x + 1},{offset_y + 1}',
        OUTPUT_ELEVATION=f'{total_input_size + 1}cm',
    ).encode('utf-8')


# 时间序列滑动窗口：每个变量一种颜色（RGB）
_TIMESERIES_COLORS = [
    (38, 139, 210),   # blue
    (42, 161, 152),   # cyan
    (133, 153, 0),    # green
    (181, 137, 0),    # orange
    (211, 54, 130),   # magenta
    (108, 113, 196),  # violet
]


def make_timeseries_tex_string(
    step: int,
    input_length: Optional[int] = None,
    kernel_size: Optional[int] = None,
    num_variables: int = 3,
    input_size: Optional[int] = None,
    stride: int = 1,
    **kwargs,
) -> bytes:
    """创建时间序列滑动窗口的 LaTeX 字符串。
    无 padding；每个变量一行，一种颜色。支持 stride（步长）。
    参数
    ----------
    step : int
        当前滑动步（输出位置索引）
    input_length : int
        时间序列长度（可与 input_size 二选一）
    kernel_size : int
        滑动窗口大小
    num_variables : int
        变量个数（每个变量一种颜色）
    input_size : int
        与 input_length 同义（兼容命令行 -i）
    stride : int
        滑动步长，默认 1
    返回
    -------
    bytes
        可被 LaTeX 编译的字符串
    """
    L = input_length if input_length is not None else input_size
    if L is None or kernel_size is None:
        raise ValueError("需要指定 input_length/input_size 与 kernel_size")
    output_length = (L - kernel_size) // stride + 1
    if step < 0 or step >= output_length:
        raise ValueError(f'step {step} 超出范围 [0, {output_length})')
    # 当前步对应的窗口在输入上的起始位置
    window_from = step * stride
    window_to = window_from + kernel_size

    template_path = Path('templates') / 'timeseries_figure.txt'
    with open(template_path, 'r', encoding='utf-8') as f:
        tex_template = f.read()

    # 每个变量一种颜色
    color_defs = []
    for v in range(num_variables):
        r, g, b = _TIMESERIES_COLORS[v % len(_TIMESERIES_COLORS)]
        color_defs.append(f'\\definecolor{{var{v}}}{{RGB}}{{{r},{g},{b}}}')
    color_defs_str = '\n    '.join(color_defs)

    # 输入层：1×1 方格
    # 输出层：长方形（宽 1、高 ROW_HEIGHT）
    row_height = 0.6
    output_height = num_variables * row_height

    # 输入网格：每格 (时间, 变量) 1×1 方格，对应变量颜色
    input_rects = []
    for t in range(L):
        for v in range(num_variables):
            input_rects.append(
                f'\\draw[draw=base03, fill=var{v}, thick] ({t},{v}) rectangle ({t+1},{v+1});'
            )
    input_rects_str = '\n    '.join(input_rects)

    # 输出层：长方形格子，每行（变量）与输入同色 var{{v}}
    output_rects = []
    for i in range(output_length):
        for v in range(num_variables):
            y0 = v * row_height
            y1 = (v + 1) * row_height
            output_rects.append(
                f'\\draw[draw=base03, fill=var{v}, thick] ({i},{y0}) rectangle ({i+1},{y1});'
            )
    output_rects_str = '\n        '.join(output_rects)

    return tex_template.format(
        COLOR_DEFS=color_defs_str,
        INPUT_RECTS=input_rects_str,
        INPUT_LENGTH=L,
        NUM_VARIABLES=num_variables,
        WINDOW_FROM=window_from,
        WINDOW_TO=window_to,
        OUTPUT_LENGTH=output_length,
        OUTPUT_RECTS=output_rects_str,
        OUTPUT_HEIGHT=output_height,
        ROW_HEIGHT=row_height,
        OUTPUT_HIGHLIGHT=step,
        OUTPUT_HIGHLIGHT_PLUS_1=step + 1,
        OUTPUT_ELEVATION=f'{num_variables + 2}cm',
    ).encode('utf-8')


def compile_figure(which_: str, name: str, step: int, compile_pdf: bool = False, **kwargs) -> bool:
    """生成 LaTeX 图文件（可选择是否编译为 PDF）。
    
    参数
    ----------
    which_ : str
        动画类型，'arithmetic'、'numerical' 或 'timeseries'
    name : str
        动画名称
    step : int
        动画步骤
    compile_pdf : bool
        如果为 True，则编译为 PDF；否则只保存 LaTeX 文件
    **kwargs
        其他参数传递给 make_*_tex_string 函数

    返回
    -------
    bool
        若请求编译 PDF 且成功则返回 True，否则（仅 LaTeX 或编译失败）返回 False。
    """
    if which_ == 'arithmetic':
        tex_string = make_arithmetic_tex_string(step, **kwargs)
    elif which_ == 'timeseries':
        tex_string = make_timeseries_tex_string(step, **kwargs)
    else:
        tex_string = make_numerical_tex_string(step, **kwargs)
    
    jobname = f'{name}_{step:02d}'
    tex_dir = Path('tex')
    tex_dir.mkdir(exist_ok=True)
    
    # 保存 LaTeX 文件
    tex_file = tex_dir / f'{jobname}.tex'
    with open(tex_file, 'wb') as f:
        f.write(tex_string)
    
    # 如果需要编译为 PDF
    if compile_pdf:
        pdflatex_cmd = _get_pdflatex_cmd()
        if not pdflatex_cmd:
            raise FileNotFoundError(
                "未找到 pdflatex。请安装 TeX Live 或 MacTeX，并确保 pdflatex 在 PATH 中。"
            )
        pdf_dir = Path('pdf')
        pdf_dir.mkdir(exist_ok=True)
        
        p = subprocess.Popen(
            pdflatex_cmd + [f'-jobname={jobname}', '-output-directory', 'pdf'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdoutdata, stderrdata = p.communicate(input=tex_string)
        
        # 如果编译成功，删除日志和辅助文件
        if b'! LaTeX Error' in stdoutdata or b'! Emergency stop' in stdoutdata:
            print(f'! LaTeX 错误: 检查 pdf/{jobname}.log 文件')
            return False
        for pattern in [f'pdf/{jobname}.aux', f'pdf/{jobname}.log']:
            for file in glob(pattern):
                os.remove(file)
        return True
    return False


# ==================== Makefile 生成部分 ====================

# 动画配置
ANIMATIONS: Dict[str, Dict[str, any]] = {
    # 时间序列滑动窗口：无 padding、stride=1，每个变量一种颜色
    'timeseries_sliding_window': {
        'type': 'timeseries',
        'input-size': 9,
        'kernel-size': 3,
        'num-variables': 3,
        'padding': 0,
        'stride': 1,
        'dilation': 1,
        'mode': 'convolution',
        'transposed': False,
    },
    # 时间序列滑动窗口（带 stride）：stride=2
    'timeseries_sliding_window_strides': {
        'type': 'timeseries',
        'input-size': 9,
        'kernel-size': 3,
        'num-variables': 3,
        'padding': 0,
        'stride': 2,
        'dilation': 1,
        'mode': 'convolution',
        'transposed': False,
    },
    # 'no_padding_no_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 4,
    #     'output-size': 2,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'arbitrary_padding_no_strides': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 6,
    #     'padding': 2,
    #     'kernel-size': 4,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'arbitrary_padding_no_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 6,
    #     'padding': 2,
    #     'kernel-size': 4,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'same_padding_no_strides': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 5,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'same_padding_no_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 5,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'full_padding_no_strides': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 7,
    #     'padding': 2,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'full_padding_no_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 7,
    #     'padding': 2,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'no_padding_strides': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 2,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'no_padding_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 2,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'padding_strides': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'padding_strides_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'padding_strides_odd': {
    #     'type': 'arithmetic',
    #     'input-size': 6,
    #     'output-size': 3,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'padding_strides_odd_transposed': {
    #     'type': 'arithmetic',
    #     'input-size': 6,
    #     'output-size': 3,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': True,
    # },
    # 'dilation': {
    #     'type': 'arithmetic',
    #     'input-size': 7,
    #     'output-size': 3,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 2,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'numerical_no_padding_no_strides': {
    #     'type': 'numerical',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'numerical_padding_strides': {
    #     'type': 'numerical',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 1,
    #     'kernel-size': 3,
    #     'stride': 2,
    #     'dilation': 1,
    #     'mode': 'convolution',
    #     'transposed': False,
    # },
    # 'numerical_average_pooling': {
    #     'type': 'numerical',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'average',
    #     'transposed': False,
    # },
    # 'numerical_max_pooling': {
    #     'type': 'numerical',
    #     'input-size': 5,
    #     'output-size': 3,
    #     'padding': 0,
    #     'kernel-size': 3,
    #     'stride': 1,
    #     'dilation': 1,
    #     'mode': 'max',
    #     'transposed': False,
    # },
}

def pdf_to_png(pdf_path: Path, png_path: Path):
    """将 PDF 文件转换为 PNG 图片。
    
    参数
    ----------
    pdf_path : Path
        PDF 文件路径
    png_path : Path
        输出 PNG 文件路径
    """
    cmd = _get_convert_cmd()
    if not cmd:
        raise FileNotFoundError(
            "未找到 ImageMagick (convert/magick)。请安装 ImageMagick 并确保在 PATH 中。"
        )
    subprocess.run(
        cmd + ['-density', '600', str(pdf_path), '-flatten', '-resize', '25%', str(png_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def pngs_to_gif(png_files: List[Path], gif_path: Path, optimize: bool = True):
    """将 PNG 图片序列转换为 GIF 动画。
    
    参数
    ----------
    png_files : List[Path]
        PNG 文件路径列表（按顺序）
    gif_path : Path
        输出 GIF 文件路径
    optimize : bool
        是否使用 gifsicle 优化 GIF（默认 True）
    """
    if not png_files:
        raise ValueError("PNG 文件列表不能为空")
    
    convert_cmd = _get_convert_cmd()
    if not convert_cmd:
        raise FileNotFoundError(
            "未找到 ImageMagick (convert/magick)。请安装 ImageMagick 并确保在 PATH 中。"
        )
    # 使用 ImageMagick 的 convert 命令创建 GIF
    cmd = convert_cmd + [
        '-delay', '100',
        '-loop', '0',
        '-layers', 'Optimize',
        '+map',
        '-dispose', 'previous'
    ] + [str(p) for p in png_files] + [str(gif_path)]
    
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 使用 gifsicle 优化 GIF
    if optimize:
        try:
            subprocess.run(
                ['gifsicle', '--batch', '-O3', str(gif_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 gifsicle 不可用，只打印警告
            print(f"  警告: gifsicle 不可用，跳过 GIF 优化")


def generate_gif_for_animation(name: str, config: Dict, pdf_dir: Path, png_dir: Path, gif_dir: Path):
    """为单个动画生成 GIF。
    
    参数
    ----------
    name : str
        动画名称
    config : Dict
        动画配置
    pdf_dir : Path
        PDF 文件目录
    png_dir : Path
        PNG 文件目录
    gif_dir : Path
        GIF 文件目录
    """
    # 计算步骤数
    if config.get('type') == 'timeseries':
        stride = config.get('stride', 1)
        steps = (config['input-size'] - config['kernel-size']) // stride + 1
    elif config.get('transposed'):
        steps = config['input-size'] ** 2
    else:
        steps = config['output-size'] ** 2
    
    # 确保目录存在
    png_dir.mkdir(exist_ok=True)
    gif_dir.mkdir(exist_ok=True)
    
    # 转换所有 PDF 为 PNG
    png_files = []
    for step in range(steps):
        pdf_file = pdf_dir / f'{name}_{step:02d}.pdf'
        png_file = png_dir / f'{name}_{step:02d}.png'
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_file}")
        
        pdf_to_png(pdf_file, png_file)
        png_files.append(png_file)
    
    # 将 PNG 序列转换为 GIF
    gif_file = gif_dir / f'{name}.gif'
    pngs_to_gif(png_files, gif_file)
    
    return gif_file


def generate_all_animations(animation_name: Optional[str] = None, compile_pdf: bool = False, generate_gif: bool = False):
    """根据配置生成所有动画的 LaTeX 文件（可选择是否编译为 PDF 或生成 GIF）。
    
    参数
    ----------
    animation_name : str, optional
        如果指定，只生成该动画；否则生成所有动画。
    compile_pdf : bool
        如果为 True，则编译为 PDF；否则只保存 LaTeX 文件（默认 False）
    generate_gif : bool
        如果为 True，则生成 GIF 动画（需要先有 PDF 文件，默认 False）
    """
    # 若请求编译 PDF，先检查 pdflatex 是否可用
    if compile_pdf and not _get_pdflatex_cmd():
        print("未找到 pdflatex，跳过 PDF 编译。")
        print("请安装 TeX Live 或 MacTeX，并确保 pdflatex 在 PATH 中。")
        compile_pdf = False
    
    # 若请求生成 GIF，检查 ImageMagick 与 PDF 来源
    if generate_gif:
        if not _get_convert_cmd():
            print("未找到 ImageMagick (convert/magick)，跳过 GIF 生成。")
            print("请安装 ImageMagick 并确保在 PATH 中。")
            generate_gif = False
        elif not compile_pdf:
            pdf_dir = Path('pdf')
            if not pdf_dir.exists():
                print("警告: 生成 GIF 需要 PDF 文件，但 pdf/ 目录不存在。")
                print("      请先运行 --compile-pdf 生成 PDF 文件，或确保 pdf/ 目录中有所需的 PDF 文件。")
                return
    
    animations_to_process = (
        {animation_name: ANIMATIONS[animation_name]}
        if animation_name and animation_name in ANIMATIONS
        else ANIMATIONS
    )
    
    total_animations = len(animations_to_process)
    current_animation = 0
    
    pdf_dir = Path('pdf')
    png_dir = Path('png')
    gif_dir = Path('gif')
    
    for name, config in animations_to_process.items():
        current_animation += 1
        print(f"[{current_animation}/{total_animations}] 正在生成动画: {name}")
        
        # 计算步骤数
        if config['type'] == 'timeseries':
            steps = (config['input-size'] - config['kernel-size']) // config['stride'] + 1
        elif config['transposed']:
            steps = config['input-size'] ** 2
        else:
            steps = config['output-size'] ** 2
        
        # 准备参数
        kwargs = {
            'input_size': config['input-size'],
            'output_size': config.get('output-size', config['input-size'] - config['kernel-size'] + 1),
            'padding': config['padding'],
            'kernel_size': config['kernel-size'],
            'stride': config['stride'],
            'dilation': config['dilation'],
            'compile_pdf': compile_pdf,
        }
        
        if config['type'] == 'numerical':
            kwargs['mode'] = config['mode']
        elif config['type'] == 'timeseries':
            kwargs['input_length'] = config['input-size']
            kwargs['kernel_size'] = config['kernel-size']
            kwargs['num_variables'] = config['num-variables']
            kwargs['stride'] = config['stride']
        else:
            kwargs['transposed'] = config['transposed']
        
        # 生成每一步，统计成功生成的 PDF 数
        pdf_ok_count = 0
        for step in range(steps):
            try:
                if compile_figure(config['type'], name, step, **kwargs):
                    pdf_ok_count += 1
            except Exception as e:
                print(f"  错误: 生成 {name} 第 {step} 步时出错: {e}")
        
        output_types = ["LaTeX"]
        if pdf_ok_count > 0:
            output_types.append("PDF")
        
        # 仅当该动画所有步骤的 PDF 都已生成时才尝试生成 GIF
        if generate_gif and pdf_ok_count == steps:
            try:
                generate_gif_for_animation(name, config, pdf_dir, png_dir, gif_dir)
                output_types.append("GIF")
            except Exception as e:
                print(f"  错误: 生成 {name} 的 GIF 时出错: {e}")
        elif generate_gif and pdf_ok_count < steps:
            print(f"  跳过 GIF: {name} 缺少 PDF 文件（仅 {pdf_ok_count}/{steps} 步成功）")
        
        print(f"  完成: {name} ({steps} 步) - 已生成 {', '.join(output_types)} 文件")
    
    print(f"\n所有动画生成完成！共 {total_animations} 个动画。")


# ==================== 主程序 ====================

def main():
    """主函数，处理命令行参数。"""
    parser = argparse.ArgumentParser(
        description="编译 LaTeX 图作为卷积动画的一部分，或生成所有动画。"
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    subparsers.required = True

    # 生成所有动画的命令
    all_parser = subparsers.add_parser(
        'all',
        help='根据配置生成所有动画的 LaTeX 文件'
    )
    all_parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="如果指定，只生成该名称的动画；否则生成所有动画"
    )
    all_parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="编译 LaTeX 文件为 PDF（默认只生成 LaTeX 文件）"
    )
    all_parser.add_argument(
        "--generate-gif",
        action="store_true",
        help="生成 GIF 动画（需要先有 PDF 文件，可与 --compile-pdf 一起使用）"
    )

    # 生成单个图的命令
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("name", type=str, help="动画名称")
    parent_parser.add_argument("step", type=int, help="动画步骤")
    parent_parser.add_argument("-i", "--input-size", type=int, default=5,
                               help="输入大小")
    parent_parser.add_argument("-o", "--output-size", type=int, default=3,
                               help="输出大小")
    parent_parser.add_argument("-p", "--padding", type=int, default=0,
                               help="零填充")
    parent_parser.add_argument("-k", "--kernel-size", type=int, default=3,
                               help="核大小")
    parent_parser.add_argument("-s", "--stride", type=int, default=1,
                               help="步长")
    parent_parser.add_argument("-d", "--dilation", type=int, default=1,
                               help="膨胀")

    arithmetic_subparser = subparsers.add_parser(
        'arithmetic',
        parents=[parent_parser],
        help='卷积算术动画'
    )
    arithmetic_subparser.add_argument(
        "--transposed",
        action="store_true",
        help="动画转置卷积"
    )
    arithmetic_subparser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="编译 LaTeX 文件为 PDF（默认只生成 LaTeX 文件）"
    )

    numerical_subparser = subparsers.add_parser(
        'numerical',
        parents=[parent_parser],
        help='数值卷积动画'
    )
    numerical_subparser.add_argument(
        "-m", "--mode",
        type=str,
        default='convolution',
        choices=('convolution', 'average', 'max'),
        help="核模式"
    )
    numerical_subparser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="编译 LaTeX 文件为 PDF（默认只生成 LaTeX 文件）"
    )

    # 时间序列滑动窗口：无 padding、stride=1，每个变量一种颜色
    timeseries_parser = subparsers.add_parser(
        'timeseries',
        help='时间序列滑动窗口（每变量一色，无 padding、stride=1）'
    )
    timeseries_parser.add_argument("name", type=str, help="动画名称")
    timeseries_parser.add_argument("step", type=int, help="滑动步（输出位置）")
    timeseries_parser.add_argument("-i", "--input-size", type=int, default=7,
                                  help="时间序列长度")
    timeseries_parser.add_argument("-k", "--kernel-size", type=int, default=3,
                                   help="滑动窗口大小")
    timeseries_parser.add_argument("--num-variables", type=int, default=3,
                                   help="变量个数（每个变量一种颜色）")
    timeseries_parser.add_argument("-s", "--stride", type=int, default=1,
                                   help="滑动步长（默认 1）")
    timeseries_parser.add_argument("--compile-pdf", action="store_true",
                                   help="编译 LaTeX 为 PDF")

    args = parser.parse_args()
    
    if args.command == 'all':
        # 生成所有动画
        generate_all_animations(
            args.name,
            compile_pdf=getattr(args, 'compile_pdf', False),
            generate_gif=getattr(args, 'generate_gif', False)
        )
    else:
        # 处理单个图生成
        args_dict = vars(args)
        which_ = args_dict.pop('command')
        name = args_dict.pop('name')
        step = args_dict.pop('step')
        compile_pdf = args_dict.pop('compile_pdf', False)
        compile_figure(which_, name, step, compile_pdf=compile_pdf, **args_dict)


if __name__ == "__main__":
    main()

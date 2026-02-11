#!/usr/bin/env python3
"""
纯时间序列滑动窗口绘图脚本。

- 依赖已有的 `templates/timeseries_figure.txt` LaTeX 模板
- 支持：
  - 普通滑动窗口（stride=1）
  - 带步长的滑动窗口（stride>1）
- 可选择只生成 `.tex`，或同时编译为 `.pdf`，并进一步生成 `.gif`

使用时请在仓库根目录下运行，例如：

    python3 new/timeseries_draw.py all --compile-pdf --generate-gif
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _get_pdflatex_cmd() -> Optional[List[str]]:
    """返回 pdflatex 命令列表，未找到时返回 None。"""
    exe = shutil.which("pdflatex")
    return [exe] if exe else None


def _get_convert_cmd() -> Optional[List[str]]:
    """返回 ImageMagick convert 命令列表（兼容 6 与 7），未找到时返回 None。"""
    exe = shutil.which("convert")
    if exe:
        return [exe]
    magick = shutil.which("magick")
    if magick:
        return [magick, "convert"]
    return None


# 时间序列滑动窗口：每个变量一种颜色（Solarized 调色板）
_TIMESERIES_COLORS = [
    (38, 139, 210),  # blue
    (42, 161, 152),  # cyan
    (133, 153, 0),   # green
    (181, 137, 0),   # orange
    (211, 54, 130),  # magenta
    (108, 113, 196), # violet
]


def make_timeseries_tex_string(
    step: int,
    input_length: int,
    kernel_size: int,
    num_variables: int = 3,
    stride: int = 1,
) -> bytes:
    """创建时间序列滑动窗口的 LaTeX 字符串（无 padding，支持 stride）。"""
    if stride <= 0:
        raise ValueError("stride 必须为正整数")

    L = input_length
    # 这里我们定义一对 (X, Y) 组合：
    # - X_i: 从位置 x_start 到 x_start + kernel_size
    # - Y_i: 紧接着的窗口，从 x_start + kernel_size 到 x_start + 2 * kernel_size
    # 因此必须满足 x_start + 2 * kernel_size <= L
    pair_steps = (L - 2 * kernel_size) // stride + 1
    if pair_steps <= 0:
        raise ValueError("input_length 与 kernel_size 组合无法形成任何 (X, Y) 对")
    if step < 0 or step >= pair_steps:
        raise ValueError(f"step {step} 超出范围 [0, {pair_steps})")

    template_path = Path("templates") / "timeseries_figure.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        tex_template = f.read()

    # 每个变量一种颜色
    color_defs = []
    for v in range(num_variables):
        r, g, b = _TIMESERIES_COLORS[v % len(_TIMESERIES_COLORS)]
        color_defs.append(f"\\definecolor{{var{v}}}{{RGB}}{{{r},{g},{b}}}")
    color_defs_str = "\n    ".join(color_defs)

    # 输入层：1×1 方格
    # 为了让输出层与输入层截取窗口“同尺寸”，输出层也使用 1×1 单位网格：
    # - 高度 = num_variables
    # - 宽度 = 2 * kernel_size（X 一段 + Y 一段）
    row_height = 1.0
    output_height = float(num_variables)

    input_rects = []
    for t in range(L):
        for v in range(num_variables):
            # 输入全部使用统一的灰色方块
            input_rects.append(
                f"\\draw[draw=base03, fill=gray!70, thick] ({t},{v}) rectangle ({t+1},{v+1});"
            )
    input_rects_str = "\n    ".join(input_rects)

    # 输出层：每一帧只展示一个 (X, Y) 组合（两个与输入窗口同尺寸的块：X 在前/左，Y 在后/右）
    output_rects = []
    # X 块（宽 kernel_size，高 num_variables，绿色）
    output_rects.append(
        f"\\draw[draw=base03, fill=green, thick] (0,0) rectangle ({kernel_size},{output_height});"
    )
    # Y 块（紧接在 X 右侧，黄色）
    output_rects.append(
        f"\\draw[draw=base03, fill=yellow, thick] ({kernel_size},0) rectangle ({2*kernel_size},{output_height});"
    )
    output_rects_str = "\n        ".join(output_rects)

    # 当前步对应的 X/Y 窗口在输入上的起始位置
    x_window_from = step * stride
    x_window_to = x_window_from + kernel_size
    y_window_from = x_window_to
    y_window_to = y_window_from + kernel_size

    # 本帧的标签：x_1, y_1, x_2, y_2, ... 以及对应维度
    label_index = step + 1
    # 顶层块：x_i \in \mathbb{R}^{T\times D}, y_i \in \mathbb{R}^{H\times D}
    x_label = (
        f"\\mathbf{{x}}_{{{label_index}}} \\in \\mathbb{{R}}^{{T\\times D}}"
    )
    y_label = (
        f"\\mathbf{{y}}_{{{label_index}}} \\in \\mathbb{{R}}^{{H\\times D}}"
    )

    # 输出层几何：两个与输入窗口同尺寸的块
    output_x_from = 0
    output_x_to = kernel_size
    output_y_from = kernel_size
    output_y_to = 2 * kernel_size
    label_x_center = kernel_size / 2
    label_y_center = kernel_size + kernel_size / 2

    # 输入层标注：长为 T，宽为 D，\mathcal{V} \in \mathbb{R}^{T\times D}
    t_label_x = L / 2.0
    t_label_y = -0.8
    d_label_x = -0.8
    d_label_y = num_variables / 2.0
    v_label_x = L
    v_label_y = num_variables + 0.3

    # 输入层中选取一个时间步 t 的位置，画箭头和 v_t \in \mathbb{R}^D
    vt_center_x = x_window_from + kernel_size / 2.0
    vt_from_x = vt_center_x
    vt_from_y = 0.0
    vt_to_x = vt_center_x
    vt_to_y = -0.4
    vt_label_x = vt_center_x + 0.3
    vt_label_y = -0.6

    # 输入层标签文本（用变量传给模板，以避免在模板中出现未转义花括号）
    v_label_text = r"\mathcal{V} \in \mathbb{R}^{T \times D}"
    vt_label_text = r"\mathbf{v}_t \in \mathbb{R}^{D}"

    return tex_template.format(
        COLOR_DEFS=color_defs_str,
        INPUT_RECTS=input_rects_str,
        INPUT_LENGTH=L,
        NUM_VARIABLES=num_variables,
        X_WINDOW_FROM=x_window_from,
        X_WINDOW_TO=x_window_to,
        Y_WINDOW_FROM=y_window_from,
        Y_WINDOW_TO=y_window_to,
        X_LABEL=x_label,
        Y_LABEL=y_label,
        # 输出层宽度固定为 2*kernel_size，仅展示一个 (X, Y)（两个块）
        OUTPUT_LENGTH=2 * kernel_size,
        OUTPUT_RECTS=output_rects_str,
        OUTPUT_HEIGHT=output_height,
        ROW_HEIGHT=row_height,
        KERNEL_SIZE=kernel_size,
        OUTPUT_X_FROM=output_x_from,
        OUTPUT_X_TO=output_x_to,
        OUTPUT_Y_FROM=output_y_from,
        OUTPUT_Y_TO=output_y_to,
        LABEL_X_CENTER=label_x_center,
        LABEL_Y_CENTER=label_y_center,
        T_LABEL_X=t_label_x,
        T_LABEL_Y=t_label_y,
        D_LABEL_X=d_label_x,
        D_LABEL_Y=d_label_y,
        V_LABEL_X=v_label_x,
        V_LABEL_Y=v_label_y,
        VT_FROM_X=vt_from_x,
        VT_FROM_Y=vt_from_y,
        VT_TO_X=vt_to_x,
        VT_TO_Y=vt_to_y,
        VT_LABEL_X=vt_label_x,
        VT_LABEL_Y=vt_label_y,
        V_LABEL_TEXT=v_label_text,
        VT_LABEL_TEXT=vt_label_text,
        # 高亮整对 (X, Y) 两个块
        OUTPUT_HIGHLIGHT=0,
        OUTPUT_HIGHLIGHT_PLUS_1=2 * kernel_size,
        OUTPUT_ELEVATION=f"{num_variables + 2}cm",
    ).encode("utf-8")


def compile_figure(
    name: str,
    step: int,
    input_length: int,
    kernel_size: int,
    num_variables: int = 3,
    stride: int = 1,
    compile_pdf: bool = False,
) -> bool:
    """生成单帧时间序列滑动窗口图（tex，必要时编译为 pdf）。"""
    tex_bytes = make_timeseries_tex_string(
        step=step,
        input_length=input_length,
        kernel_size=kernel_size,
        num_variables=num_variables,
        stride=stride,
    )

    jobname = f"{name}_{step:02d}"
    tex_dir = Path("tex")
    tex_dir.mkdir(exist_ok=True)

    tex_file = tex_dir / f"{jobname}.tex"
    tex_file.write_bytes(tex_bytes)

    if not compile_pdf:
        return False

    pdflatex_cmd = _get_pdflatex_cmd()
    if not pdflatex_cmd:
        raise FileNotFoundError(
            "未找到 pdflatex。请安装 TeX Live 或 MacTeX，并确保 pdflatex 在 PATH 中。"
        )

    pdf_dir = Path("pdf")
    pdf_dir.mkdir(exist_ok=True)

    p = subprocess.Popen(
        pdflatex_cmd + [f"-jobname={jobname}", "-output-directory", "pdf"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdoutdata, _ = p.communicate(input=tex_bytes)

    if b"! LaTeX Error" in stdoutdata or b"! Emergency stop" in stdoutdata:
        print(f"! LaTeX 错误: 检查 pdf/{jobname}.log 文件")
        return False

    # 清理辅助文件
    for suffix in (".aux", ".log"):
        aux = pdf_dir / f"{jobname}{suffix}"
        if aux.exists():
            aux.unlink()

    return True


def pdf_to_png(pdf_path: Path, png_path: Path) -> None:
    """将单页 PDF 转为 PNG。"""
    cmd = _get_convert_cmd()
    if not cmd:
        raise FileNotFoundError(
            "未找到 ImageMagick (convert/magick)。请安装 ImageMagick 并确保在 PATH 中。"
        )
    subprocess.run(
        cmd
        + [
            "-density",
            "600",
            str(pdf_path),
            "-flatten",
            "-resize",
            "25%",
            str(png_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def pngs_to_gif(png_files: List[Path], gif_path: Path) -> None:
    """将 PNG 序列合成为 GIF。"""
    if not png_files:
        raise ValueError("PNG 文件列表不能为空")

    convert_cmd = _get_convert_cmd()
    if not convert_cmd:
        raise FileNotFoundError(
            "未找到 ImageMagick (convert/magick)。请安装 ImageMagick 并确保在 PATH 中。"
        )

    cmd = convert_cmd + [
        "-delay",
        "100",
        "-loop",
        "0",
        "-layers",
        "Optimize",
        "+map",
        "-dispose",
        "previous",
    ] + [str(p) for p in png_files] + [str(gif_path)]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 尝试用 gifsicle 进一步优化（可选）
    gifsicle = shutil.which("gifsicle")
    if gifsicle:
        subprocess.run(
            [gifsicle, "--batch", "-O3", str(gif_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def generate_gif_for_animation(
    name: str,
    input_length: int,
    kernel_size: int,
    num_variables: int,
    stride: int,
) -> Path:
    """根据已生成的 PDF 序列，为一个时间序列动画生成 GIF。"""
    # 与 make_timeseries_tex_string 中的 pair_steps 保持一致
    steps = (input_length - 2 * kernel_size) // stride + 1

    pdf_dir = Path("pdf")
    png_dir = Path("png")
    gif_dir = Path("gif")
    png_dir.mkdir(exist_ok=True)
    gif_dir.mkdir(exist_ok=True)

    png_files: List[Path] = []
    for step in range(steps):
        pdf_file = pdf_dir / f"{name}_{step:02d}.pdf"
        png_file = png_dir / f"{name}_{step:02d}.png"
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_file}")
        pdf_to_png(pdf_file, png_file)
        png_files.append(png_file)

    gif_file = gif_dir / f"{name}.gif"
    pngs_to_gif(png_files, gif_file)
    return gif_file


# 预设的纯时序动画配置
TIMESERIES_ANIMATIONS: Dict[str, Dict[str, int]] = {
    "timeseries_sliding_window": {
        "input_length": 9,
        "kernel_size": 3,
        "num_variables": 3,
        "stride": 1,
    },
    # "timeseries_sliding_window_strides": {
    #     "input_length": 7,
    #     "kernel_size": 3,
    #     "num_variables": 3,
    #     "stride": 2,
    # },
}


def generate_all_animations(
    name: Optional[str] = None,
    compile_pdf: bool = False,
    generate_gif_flag: bool = False,
) -> None:
    """批量生成纯时间序列动画。"""
    if name:
        if name not in TIMESERIES_ANIMATIONS:
            raise ValueError(f"未知动画名称: {name}")
        animations = {name: TIMESERIES_ANIMATIONS[name]}
    else:
        animations = TIMESERIES_ANIMATIONS

    total = len(animations)
    current = 0

    for anim_name, cfg in animations.items():
        current += 1
        print(f"[{current}/{total}] 生成动画: {anim_name}")

        input_length = cfg["input_length"]
        kernel_size = cfg["kernel_size"]
        num_variables = cfg["num_variables"]
        stride = cfg["stride"]

        # 与 make_timeseries_tex_string 中的 pair_steps 保持一致
        steps = (input_length - 2 * kernel_size) // stride + 1

        pdf_ok = 0
        for step in range(steps):
            try:
                if compile_figure(
                    name=anim_name,
                    step=step,
                    input_length=input_length,
                    kernel_size=kernel_size,
                    num_variables=num_variables,
                    stride=stride,
                    compile_pdf=compile_pdf,
                ):
                    pdf_ok += 1
            except Exception as e:
                print(f"  错误: 生成 {anim_name} 第 {step} 步时出错: {e}")

        out_types = ["tex"]
        if pdf_ok > 0:
            out_types.append("pdf")

        if generate_gif_flag and compile_pdf and pdf_ok == steps:
            try:
                generate_gif_for_animation(
                    anim_name, input_length, kernel_size, num_variables, stride
                )
                out_types.append("gif")
            except Exception as e:
                print(f"  错误: 生成 {anim_name} 的 GIF 时出错: {e}")

        print(f"  完成: {anim_name} ({steps} 步) - 已生成 {', '.join(out_types)} 文件")

    print(f"\n所有时间序列动画生成完成！共 {total} 个动画。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="纯时间序列滑动窗口绘图脚本（生成 tex/pdf/gif）。"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    subparsers.required = True

    # 批量生成
    all_parser = subparsers.add_parser(
        "all", help="根据预设配置批量生成时间序列动画"
    )
    all_parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="只生成指定动画（默认生成所有预设动画）",
    )
    all_parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="编译 tex 为 pdf",
    )
    all_parser.add_argument(
        "--generate-gif",
        action="store_true",
        help="在已生成 pdf 的基础上输出 gif 动画",
    )

    # 单帧生成
    ts_parser = subparsers.add_parser(
        "timeseries", help="生成单帧时间序列滑动窗口图"
    )
    ts_parser.add_argument("name", type=str, help="动画名称（仅用于文件前缀）")
    ts_parser.add_argument("step", type=int, help="滑动步（输出位置索引）")
    ts_parser.add_argument(
        "-i", "--input-size", type=int, default=7, help="时间序列长度"
    )
    ts_parser.add_argument(
        "-k", "--kernel-size", type=int, default=3, help="滑动窗口大小"
    )
    ts_parser.add_argument(
        "--num-variables", type=int, default=3, help="变量个数（每个变量一种颜色）"
    )
    ts_parser.add_argument(
        "-s", "--stride", type=int, default=1, help="滑动步长（默认 1）"
    )
    ts_parser.add_argument(
        "--compile-pdf", action="store_true", help="编译 tex 为 pdf"
    )

    args = parser.parse_args()

    if args.command == "all":
        generate_all_animations(
            name=args.name,
            compile_pdf=getattr(args, "compile_pdf", False),
            generate_gif_flag=getattr(args, "generate_gif", False),
        )
    else:
        # 单帧
        ok = compile_figure(
            name=args.name,
            step=args.step,
            input_length=args.input_size,
            kernel_size=args.kernel_size,
            num_variables=args.num_variables,
            stride=args.stride,
            compile_pdf=args.compile_pdf,
        )
        kind = "tex+pdf" if ok and args.compile_pdf else "tex"
        print(f"已生成 {kind}：tex/{args.name}_{args.step:02d}.tex")


if __name__ == "__main__":
    main()


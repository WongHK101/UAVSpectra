# 跨项目评测参考包（2026-09-22）

这是当前 GS-GCP/UAVGS 评测实现的**源码参考快照**，不是已经适配任意数据集的通用CLI，
也不是新的科学协议或GPU实验授权。优先阅读本页，再读仓库根README和历史协议。
本包不包含真实图像、GCP/LiDAR数据、模型、密钥、上游训练仓库、指标权重或901关机控制器。

## 入口与三种不同的几何指标

下列路径均相对于完整仓库或解压后的 `GS-GCP-evaluation-reference/`。

| 内容 | 实现 | 输入与含义 |
|---|---|---|
| GCP检查点精度 | `code/gcp/evaluate_m3m_native_quarter_geometry.py` | 固定检查点、相机、像素观测和metric-depth包；检查点RMSE，不是Mesh F-score |
| RGB质量 | `code/gcp/evaluate_m3m_native_quarter_rgb_quality.py` | 固定留出GT与同尺寸渲染PNG，统一PSNR/SSIM/LPIPS |
| 渲染深度表面→LiDAR | `code/gcp/evaluate_uavgs_lidar_rectangular_roi_v1.py` | depth packet或已经明确绑定原点的表面点；训练视图/留出视图必须分别标记 |
| 原生Mesh→LiDAR | `reuse/evaluation_reference_v1/score_native_mesh.py` | 原生后处理三角Mesh；精确ROI裁剪→按面积采样→体素去重→双向最近邻 |
| 共用体素与最近邻算子 | `code/gcp/evaluate_m3m_gcp_lidar_formal_v1.py` | float64局部坐标、编码/去重、距离和P/R/F统计；其旧ROI构造入口不等于当前矩形ROI轨 |
| 深度语义 | `code/gcp/metric_depth_packet.py` | alpha归一化期望camera-z，即 M1/A；不能替换为射线距离、未归一化量或调和深度 |

Mesh评分的实际组合是 `native_mesh/mesh_score.py` 加 `native_mesh/exact_voxel_union.py`，
再调用矩形ROI与共用最近邻核心。两份Mesh文件与901已运行版本**逐字节相同**，SHA见
`PROVENANCE.json`。精确并集只调整集合合并顺序，不改变采样随机流、体素或指标。
Gaussian中心点PLY不是三角Mesh；高斯彩色全览、深度表面F10、Mesh F10不可混称。

本包原样保留的Mesh输出解释为“原生后处理Mesh”。辅助入口会拒绝将raw Mesh冒充post；
新项目若比较raw/post，应建立明确区分表示类型的两个协议/输出，并保留原版作为参照。

## 安全的首次验证（不读取真实实验数据）

在解压目录中执行，路径在Windows/Linux均可使用。第一条只用Python标准库：

```text
python -B reuse/evaluation_reference_v1/validate_reference.py --root . --verify-only
python -B reuse/evaluation_reference_v1/validate_reference.py --root . --smoke
```

第二条需要真实依赖；它不使用缺失依赖的stub，不导入torch、不访问网络、不启动CUDA，
只测试小型数组：裁剪、面积采样、原/优化并集一致、F5/F10边界、坐标归一化逆变换，
以及已有CPU depth-packet和ROI回归。它也检查GCP/RGB评测模块可导入。
**通过不代表真实GCP/LiDAR评分、LPIPS权重、GPU导出或新数据集有效性已验证。**

只查看Mesh调用信息，不评分：

```text
python -B reuse/evaluation_reference_v1/score_native_mesh.py --plan PLAN.json --method METHOD --mesh POST.ply --output NEW_OUTPUT
```

真实评分还须显式 `--execute --acknowledge-benchmark-specific-coordinate-contract`。
这两个标记只避免误调用，不会自动适配坐标，也不构成任何云端算力授权。
先在新项目中完成下一节的协议适配与独立小样本核验，再考虑大数据评分。

## 新项目必须重新确定，不能照抄的内容

1. **坐标系/高程**：当前Mesh代码写有 EPSG:4545→EPSG:32649；当前LiDAR代码使用
   `normal_minus_ellipsoid_m=23.980600991639484`。这是本数据的正常高/椭球高处理，
   不是通用常数。Mesh经当前Sim3已是正常高，不能再加一次LiDAR高程改正。
   新项目应在自己的命名空间配置水平CRS、单位、垂直基准、变换方向，并用独立控制点验证。
   **现有矩形评分器部分量是代码常量，只改JSON并不足以完成跨数据集适配。**
2. **Sim3与模型归一化**：固定控制点估计配准，独立检查点报告；不可用检查点或LiDAR
   调参/挑模型/重拟合以提高评分。必须同时逆转归一化平移与尺度。
3. **相机/像素与划分**：PINHOLE内参、图像域、零基像素中心、camera-z语义、train/holdout
   名单和SHA要重新绑定。当前native-quarter尺寸/场景名/相机数不是其他数据的要求。
4. **ROI与参考缓存**：当前是由图像覆盖确定的矩形、无垂直裁剪；不要按算法结果挑ROI。
   新ROI、原点、LiDAR清单、Sim3变化后必须新建绑定的参考缓存，不改旧缓存JSON掩盖变化。
5. **评分口径**：本项目采样密度/种子来自冻结score plan，评分体素5cm，P/R/F阈值
   5/10/20cm，边界epsilon为1e-9m。新项目应预注册自己的值；所有方法统一，并记录覆盖率。
   采样密度不是顶点数，表面孔洞不通过补洞或筛掉困难区域来隐藏。
6. **RGB指标**：统一GT、分辨率、PNG量化路径与指标实现；不要每个方法调用不同的官方
   metric脚本后直接混表。LPIPS需要匹配的参考实现及VGG/LPIPS权重SHA，本包没有这些权重。
7. **声明与权限**：本包的历史configs是协议示例，里面的场景、路径、SHA、registry不是新项目
   的授权。保留来源/改动diff；检查复制代码及上游组件的许可，不把本参考包视作额外许可证。

## 依赖与来源

- `requirements-lidar-observed.txt` 记录本次真实901评分解释器（Python3.12）的已观测版本，
  **不是一键安装承诺或跨平台完整锁文件**；不含平台、系统库、LAZ解码后端或RGB完整环境。
- GCP/RGB元数据与CPU检查还用Pillow；完整RGB指标另需匹配torch、torchvision、
  指标参考仓库/LPIPS实现和权重。原生模型导出另需各方法的合格CUDA适配器，不在本包内。
- 原始公共算子来自Git `52df095d6df61e738c5f9b4b43f076bee289a462`。新增参考封装和Mesh代码
  原先未由该commit覆盖，所以以 `REFERENCE_MANIFEST.json` 的**逐文件SHA**为完整快照身份。
- `PROVENANCE.json` 绑定Mesh来源和原批次归档。根目录Git历史和历史运行文档不等于当前授权。
- `M3M_GCP_LIDAR_FORMAL_PROTOCOL_V1.md` 用于了解共用核心/历史约定；当前矩形ROI以
  `evaluate_uavgs_lidar_rectangular_roi_v1.py` 和 `uavgs_image_defined_rectangular_roi_v1.json` 为准。

## 在本仓库重新生成参考包

```text
python -B reuse/evaluation_reference_v1/build_reference_bundle.py --repo . --output NEW_EMPTY_DIRECTORY_OUTSIDE_REPO
```

生成器按评测入口的本地Python依赖闭包选择源码，加入必要配置/文档、此参考入口、SHA清单与zip。
它拒绝覆盖已有输出；不会调用SSH、Git提交、训练、真实评分或关机。解压包只是选定评测源码集，
不能当完整基准仓库或直接用于所有正式launch gate（例如Git身份、registry和数据release检查）。

借鉴顺序建议：先完整保留本参考快照→在新项目建立独立坐标/数据协议→用合成与已知小样本
验证→固定配置和输出表示→最后运行真实数据。不要把新项目改动回写到本项目冻结的评测核心。

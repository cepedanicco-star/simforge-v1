# 安装、数据契约与原生复现

## 环境分开

离线工具用Python3.12，先安装requirements.txt，再`pip install --no-deps -e .`。requirements.txt固定核心和开发依赖；原生SDK不在这份文件中。测试与快检不需要网络服务凭据。

原生验证基线为Windows、UE5.4、Isaac Sim4.5.0、Isaac Lab v2.1.0、原生Python3.10与Torch2.5.1。Isaac/UE首次安装、GPU驱动、扩展下载和美术资产准备可能远超30分钟。使用供应商的安装方式，不要把offline usd-core安装进Isaac。configs/requirements-native-constraints.txt只记录原生集成使用过的版本约束，不是代替SDK安装器的完整锁文件。可选h5py/ONNX overlay只有确有二进制兼容需求时才配置。

所有示例从源码仓库根目录执行。已填路径和参数保存为`configs/*.local.yaml`，源模板保留不动；`local/`放使用者自己合法取得的资产。路径按当前工作目录解析。`<LOCAL_...>`为待填写项，不能原样运行。

## 许可资产与映射

必须自备公寓USD/纹理、G1带Dex3手部的机器人配置与资产、匹配厂商腿部ONNX、历史冻结BC（如要复现旧策略）、路径和桌面规划。它们没有随源码提供。机器人需具备手指和真实关节驱动；不能用无手版本替代抓取。

场景的杯路径、电脑桌路径、G1关节名和cam01须与配置/原生worker一致。新场景先生成manifest和审阅object_id/prim_path映射。默认不能自动把所有网格变成刚体，也不能自动凭类别断言摩擦系数、质量或惯量真实。

## UE导出与装配

在UE中开启Epic USD Importer和Python Editor Script Plugin，加载要导出的完整区域。编辑器Python控制台将仓库的pipeline/ue目录加入sys.path，随后：

```python
import export_scene
export_scene.export_scene('<LOCAL_EXPORT_OUTPUT>')
```

导出器烘焙材质并记录已加载组件/实例映射；还要人工检查未加载单元、USD缺失引用、cam01的时间码与镜头属性。`pipeline/ue/audit_scene.py`用于补充UE侧检查。外部普通Python不能导入unreal。

普通离线Python可执行：

```bash
python -m pipeline inspect local/scene.usda
python -m pipeline assemble local/scene.usda --out runs/prepared-001 --manifest local/manifest.json --registry local/profiles.json
```

manifest/profiles契约来自原导出/审核过程，schema位于configs/schemas。只有明确允许使用示例/估计值的开发预览才加`--allow-provisional`；此选项不会授予“训练环境已标定”的结论。输出里的plan.json可作为M1 label_plan，之后仍需审查关键目标标签。

## 材质与同机位检查

`python -m pipeline materials-audit --help`、`materials-repair --help`、`materials-refine --help`提供材质审计与新意见层修订。`materials-seven --source local/Apartment_sim.usda --out runs/materials-001`是原公寓命名专用的七项修订配方；要求真实目标shader/贴图存在，不能用于任意场景。`desk-normal`单独处理木桌。每次输出都用新目录，禁止覆盖源贴图或原层。

`material-pair-ue`要在UE环境执行，并设置AISIM_MATERIAL_CONFIG指向本地夹具JSON配置；`material-pair-isaac`在原生Isaac中执行。夹具包含相同相机矩阵、朝向、镜头、曝光和光源定义。相机配对检查是统一函数，先过门禁再渲染；先做灰卡照明隔离，再解释材质ΔE/SSIM。配对脚本不生成真实材料常数。M3的sensor_config相对随机化配置所在目录解析；其余资产路径按仓库工作目录解析。

配对/量化入口如下；先生成自己场景的夹具，不能直接重用历史公寓矩阵。夹具至少包含source_scene、output、views[6]、lights、resolution；每个view含source_frame、usd_camera_matrix、ue_position_cm、horizontal_fov_degrees、vertical_fov_degrees、lens。UE采集须在工程副本内执行。原公寓配方依赖目标shader命名，并不保证任意资产能自动匹配。

```bash
python -m pipeline.native.accept_material_pairing --config local/material-fixture.json
python -m pipeline.native.run_material_capture --config local/material-fixture.json --ue-acceptance runs/material-001/camera-pairing-acceptance.json --native-python <LOCAL_NATIVE_PYTHON>
python -m pipeline.native.quantify_material_alignment --config local/material-fixture.json --ue-acceptance runs/material-001/camera-pairing-acceptance.json
```

灰卡使用单独的标定夹具：views只含新增标定机位，calibration_cards包含gray18/white90的reflectance、position_m、dimensions_m，exposure含knob、baseline_seconds、max_trials。knob固定为`/rtx/post/tonemap/exposureTime`；灰卡位置满足该模块的正视投影约束，覆盖至少30%。先在UE副本运行capture_graycard_ue.py，再用`python -m pipeline.native.measure_graycard_ue --config local/gray-fixture.json`核验。随后给run_material_capture加`--graycard-config local/gray-fixture.json`，原生模块只拟合曝光时间，保存每次试验与灰/白残差，锁定后采6个原始机位。使用新的output目录，不覆盖基线。

新曝光采集完成后执行`python -m pipeline.native.quantify_v2_graycard --root runs/material-calibrated-001 --baseline runs/material-001`，需root内有fixture-config.json与calibration/calibration-report.json（可由使用者从实际标定目录复制同一收据，保留哈希）。该入口比较标定前后ΔE与整幅SSIM；白卡不满足时单独标注线性归一化对照，不能把归一化图片当成原生渲染。原始quantify_material_alignment同时保留历史“完整窗口均位于掩码内”的SSIM字段；最终跨曝光对照以ssim_full_frame为准。

## M1：原生采集

复制configs/sensors_default.yaml为sensors.local.yaml，填scene、label_plan、output、demo.ffmpeg，检查cam01/杯/桌/G1路径。原生Python执行：

```bash
python -m pipeline capture --config configs/sensors.local.yaml
```

20帧会写rgb.png、depth.npy、semantic.png、instance.png、normals.npy、boxes_2d.json、boxes_3d.json、camera.json、labels.json及原生annotation/overlay。深度为image-plane米制距离；原生语义/实例ID映射与PNG映射一起记录。2D边缘误差≤5px、主点偏差≤2px、关键目标必须有标签，源层哈希不变。某个必需通道空就停，不从USD估算3D框补齐。

## M2：固定抓取扫参

填sweep_friction.yaml的本地副本，source_recording必须是完整原生录制目录，`legacy_source`填仓库根目录（原命名保留，供原生worker导入pipeline包）。填写匹配的leg_policy、原生Python、临时/portable目录和必要overlay。输出不得指向已验收目录。

```bash
python -m pipeline m2 --config configs/sweep.local.yaml --preflight-only
python -m pipeline m2 --config configs/sweep.local.yaml
```

前者仅原生预检，不等于45格验收。正式为9格×5seed；抬升≥50mm、前0.5s滑移≤5mm、保持≥2s，同seed精确复现。输出raw.csv、热图、reproducibility和汇总；这些是使用者本地实验数据，不随本仓分发。历史网格2/45不能当作新策略成功率，也不能推导通用临界摩擦。

## M3：采样、采集、验收

复制randomize_office_v03.yaml为randomize.local.yaml，填source_recording、lighting_calibration、原生runtime和输出路径；sensor_config指向已填的M1副本。

```bash
python -m pipeline m3-sample --config configs/randomize.local.yaml --out runs/m3-001
python -m pipeline m3-capture --config configs/randomize.local.yaml --root runs/m3-001
python -m pipeline m3-sample --config configs/randomize.local.yaml --out runs/m3-repeat-001
python -m pipeline m3-capture --config configs/randomize.local.yaml --root runs/m3-repeat-001
python -m pipeline m3-sample --config configs/randomize.local.yaml --disable D6 --out runs/m3-no-physics-001
python -m pipeline m3-capture --config configs/randomize.local.yaml --root runs/m3-no-physics-001
python -m pipeline m3-accept --root runs/m3-001 --repeat-root runs/m3-repeat-001 --ablation-root runs/m3-no-physics-001 --out runs/m3-audit-001 --ffmpeg <LOCAL_FFMPEG>
```

完整验收需原生框非空、同seed采样值/相机/原生框几何一致（不承诺随机渲染噪声逐像素一致）、D6关闭只影响一维，以及0/5两个变体的视觉对比。验收输出两图对照及可选5秒视频，数值通过后仍应人工查看。发布版验收器使用显式repeat/ablation目录，不依赖旧v01失败现场。`capture_m3 --reuse-v01-root`仅用于已有且核验过的旧样本，不随包提供这些样本。相机净空阈值保持0.15m；缺live物理查询时必须显式提供真实表面距离实现，缺依赖则拒绝。

## M4：整合与COCO

复制pipeline_office.yaml为pipeline.local.yaml，指定真实的m1_data/m2_data/m3_data/m3_acceptance/baseline_config/sensor_config/scene/runtime。先运行本机pytest，把真实passed/total计数写入M4输出目录的regression.json；不能复制历史测试数。只有passed==total的收据才能生成M4正式报告。

```bash
python -m pytest -q
python -m pipeline m4 run --config configs/pipeline.local.yaml --scene apartment --baseline apartment
```

M4核验缓存原始哈希、重新采集指定原生帧、合并COCO并检查掩码逐像素一致。历史26图/703实例只属于旧数据；新资产应报告新计数。旧失败目录保留，新输出重跑不能伪造“缓存通过”。

## BC数据、训练与闭环

`training.extract`需要至少一条训练轨迹和一条独立验证轨迹，原始目录包含physics-body-states.npz、probe-report.json、独立物理审计，以及以冻结14关节顺序给出的上下界。`--cup-center-local`来自目标资产的不可变局部几何中心，不是用估算框替代原生annotation。

```bash
python -m training.extract --train local/demo-train-01 --validation local/demo-val-01 --joint-limits local/joint-limits.json --cup-center-local 0 0 0.05 --out runs/bc-data/demos.npz
python -m training.train --config configs/bc_training.yaml --dataset runs/bc-data/demos.npz --out runs/bc-new-001
```

上面的0/0/0.05只是占位示例，必须换成实际局部中心。提取的是50维观测、8×14相对动作和有效mask，拒绝重复轨迹/交叉分组；新数据与旧875行不同，不声称位级重现历史训练。可选合成样本只能训练使用且须额外碰撞验证。未收敛退出码2并保留新模型诊断，不能用于正式推理。

复制runtime.example.yaml和expert-arguments.example.yaml为本地副本，填写所有许可资产及原生Python，指定新run_root/output。纯BC、增强执行、专家、场景预检分别运行：

```bash
python -m training.evaluate --config configs/runtime.local.yaml --mode bc --seed 1 --dry-run
python -m training.evaluate --config configs/runtime.local.yaml --mode bc --seed 1
python -m training.evaluate --config configs/runtime.local.yaml --mode safe --seed 1
```

每次必须更换output，模式`expert`不计BC成功；`preflight`是历史v2布局提案检查，不启动BC，G0未过就停。一次命令只评一回合，不自动重试挑成功。完整20seed统计需保存每个seed初始化可行性、真实调用数、物理结果和harness状态，分别报条件/工作流分母；零调用不能记为策略失败。

## 专家审计和成片

专家通道用原生`--mode expert`，须提供已有路径/桌面规划与厂商策略。成功后，依次：

```bash
python -m pipeline.native.audit_apartment_recording --recording runs/expert/physics --placement local/placement-plan.json
python -m pipeline.native.audit_apartment_forward_motion --recording runs/expert/physics --out runs/expert-motion-audit
python -m pipeline render --recording runs/expert/physics --out runs/movie-001 --duration 25 --preserve-source-fov --ffmpeg <LOCAL_FFMPEG> --ffprobe <LOCAL_FFPROBE>
```

原生渲染使用2048×1152和25fps，625帧成25秒片。审计先验证杯桌接触、释放、稳定、身体朝向/前向位移与不允许的碰撞。重放工程Measured_replay.usdc供本地Isaac人工打开；视频重放关闭物理，源物理过程独立验证。源动作与视频时间不同要记录倍率。新源轨迹没有成功审计时不可作为成功成片。

## 复现完成度

本地发布验收会验证依赖安装、离线快检、核心算法、训练器测试、打包后的导入和清洗；不重新产生原公寓物理成功或声称新机器原生全部PASS。原生SDK/美术资产/模型不分发，完整复现实验由使用者按以上门禁运行。未经测量项始终保持null/NOT_RUN。

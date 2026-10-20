# UE → Isaac Pipeline · v1.0.0

将 UE 场景导出为 USD，在 Isaac Sim 中装配物理环境、采集原生传感器数据，训练/评估手臂 BC，并渲染经审计的机器人搬杯演示。**首版是可检查的源码工作流，不是附带公寓、机器人资产和权重的一键训练镜像。**

[在线观看专家演示](https://ue-isaac-portfolio.app.workbuddy.host/#expert-demo) · [下载25秒视频](https://ue-isaac-portfolio.app.workbuddy.host/assets/expert-transfer-plan25.mp4) · [架构](docs/ARCHITECTURE.md) · [方法与失败边界](docs/METHOD.md) · [各阶段demo与量化门禁](docs/DEMO_ACCEPTANCE.md) · [首版说明](RELEASE_v1.md)

## 先看真实结果

| 通道 | 实测结果 | 口径 |
|---|---|---|
| 原公寓 v1 的特权 BC | **0/12**，241次真实推理；7倾倒/4碰撞/1失衡 | 20seed中初始化放行12个；其余8个策略结果null；全工作流0/20 |
| v1 + 特权伺服 + 原生物理过滤 | plan24两轮各**0/3**，278次真实推理 | 正式12/20评测未运行，正式成功率null |
| 场景可行性工程提案 v2 | **G0 0/3**，0个策略回合 | 包含承托台面的整体杯—橱柜距离≥8cm约束未满足；分布已变，不能回写v1 |
| 独立专家演示 | **25s / 625帧 / 2048×1152 / 25fps** | 58.72s源物理轨迹以2.3488×重放；源物理和前向行走审计通过；**不是BC单独成功** |
| 新场景 / PPO | 尚未执行，成功率null | PPO热启动为下一步研究计划，不是现有首版成绩 |

“不可行角落”是原v1困难布局的诊断标签，不证明所有控制器都不可行：名义专家能够完成任务。当前优化执行栈的准确定性是 **冻结BC + 特权伺服 + 原生物理过滤**。它用50维真实状态；不是端到端视觉抓取，也没有新训练的全身搬杯RL策略。

历史checkpoint_p5的SHA256：`05d75af2808edfd679e9d4d4a856fa5f2fba3ac29755efcfd70898fd97485a23`。旧验证RMSE约0.03215rad不等于任务成功率。plan23–25没有重训，v10只是报告版本。仅清洗后的文字/汇总指标见[results](results/SUMMARY.md)。

## 30分钟内的资产无关快检

适用于Python3.12、可联网安装依赖的普通开发机；不需要UE、GPU、公寓或checkpoint。以下会真实生成/装配一个自有合成USD并检查坐标、种子和动作边界，**不冒充原生M1–M4或机器人成功**。

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pipeline smoke
.venv\Scripts\python -m pytest -q
```

Linux/macOS将`.venv\Scripts\python`替换为`.venv/bin/python`；原生机器人部分只在历史Windows配置中验证过。快检默认在临时目录工作；显式保留demo可用`python -m pipeline smoke --out runs/smoke-001`。每次使用新目录。

完整公寓复现需要预先安装SDK并取得许可资产，不能保证从零安装UE/Isaac后30分钟完成。首版没有附这些文件，也不以离线测试替代GPU验收。依赖、准备清单和原生命令见[运行手册](docs/RUNBOOK.md)。

## 完整链路入口

在已经安装UE5.4、Isaac Sim4.5、Isaac Lab2.1.0以及许可G1/Dex3资产的环境中，从仓库根目录执行。UE脚本用编辑器自带Python；原生仿真命令用Isaac的Python3.10，**不要在该环境装offline的usd-core**。

1. **导出**：UE启用USD Importer/Python Editor Script Plugin，使用`pipeline/ue/export_scene.py`导出已加载区域并烘焙材质；不要把未加载单元或未烘焙实例当成完整场景。
2. **装配**：`python -m pipeline assemble local/scene.usda --out runs/prepared-001 --manifest local/manifest.json --registry local/profiles.json`。建筑静态、可移动杯子动态；语义映射和材料参数来源单独审核，未知值不伪称实测。
3. **M1**：填`configs/sensors_default.yaml`的本地副本后，`python -m pipeline capture --config configs/sensors.local.yaml`。20帧RGB/深度/标签/原生框及相机内外参，门禁与文件名见手册。
4. **M2**：`python -m pipeline m2 --config configs/sweep.local.yaml`。摩擦0.3/0.6/0.9 × 质量0.2/0.5/1.0kg × 5seed，共45次；严格抬升/滑移/保持条件不变。
5. **M3**：`m3-sample`→`m3-capture`→`m3-accept`；6变体、同seed复现、D6单维消融、2个变体视觉对照。相机真实表面净空固定0.15m，不退化为整个网格AABB。
6. **M4**：配置已完成的M1/M2/M3输出，再`python -m pipeline m4 run --config configs/pipeline.local.yaml --scene apartment --baseline apartment`。生成COCO、整合报告；只能复用核验过的缓存，不能把未运行项补成通过。
7. **BC训练**：用`python -m training.extract --help`从多条经物理审计的专家轨迹导出50维观测和K8动作块；按完整轨迹划分训练/验证。`python -m training.train --config configs/bc_training.yaml`训练新模型；不覆盖历史权重。
8. **闭环评测**：先填`configs/runtime.example.yaml`的本地副本，用`python -m training.evaluate --config configs/runtime.local.yaml --mode bc --seed 1 --dry-run`审阅，再去掉`--dry-run`。`safe`为BC+伺服/过滤，`expert`独立记账；`preflight`重查历史v2提案，预期G0失败并停止。
9. **渲染**：先通过`pipeline/native/audit_apartment_recording.py`和`audit_apartment_forward_motion.py`，再`python -m pipeline render --recording runs/expert/physics --out runs/movie-001 --duration 25 --preserve-source-fov --ffmpeg <LOCAL_FFMPEG> --ffprobe <LOCAL_FFPROBE>`。

## 目录

```text
README.md / RELEASE_v1.md / LICENSE / NOTICE.md
requirements.txt          Python3.12离线工具的固定依赖
docs/                     ARCHITECTURE、METHOD、RUNBOOK、公开步骤与许可归属
pipeline/                 UE导出、USD装配、材质、M1–M4、原生worker与渲染
training/                 数据提取、50维BC训练/推理、伺服/过滤、闭环评测
configs/                  脱敏配置模板、物理schema、原生环境约束
results/                  仅清洗后的结论文字与汇总CSV
tests/                    离线算法/契约/打包回归
```

旧的图像BC架构、空场景课程、个人面试资料、数十份plan和实验时限脚本已从本发布树排除。运行产生的JSON/日志、USD、图像、权重只写本地忽略目录；原始实验JSON和逐拍账本不进入发布包。COCO源码目录正常保留，忽略规则只排除根目录的运行数据。

## 许可与发布状态

本项目原创代码MIT；Unitree观测/动作适配保留Apache-2.0，见[NOTICE](NOTICE.md)。UE公寓、纹理、NVIDIA/Unitree资产和厂商权重须另行依法取得；MIT不授予它们的再分发权。

这是全新单根历史的本地v1仓库，`v1.0.0`仅在本地标记。**待用户公开：没有自动创建GitHub Release、没有改可见性、没有覆盖旧远程历史。** 已有私有simforge仍含旧实验历史，不能因为本地包干净就直接公开旧仓。[正确的手动上传/公开步骤](docs/PUBLISH_MANUALLY.md)。

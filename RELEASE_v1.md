# UE→Isaac Pipeline v1.0.0 · 首版发布说明

状态：**本地发布候选包就绪，待用户手动公开。没有自动改GitHub可见性，没有创建GitHub Release。**

## 功能

- UE5.4 USD导出/材质烘焙、组件/语义映射、来源明确的静态/动态物理装配。
- 七项公寓材质修订配方、相机方向/FOV统一门禁、灰卡曝光与材质差异隔离工具。
- M1原生RGB-D/标签/检测框，M2摩擦质量扫参，M3六维随机化，M4 CLI/COCO/来源核验。
- 50维特权BC训练/推理、完整轨迹划分检查、K8/h2动作分块、独立闭环评测。
- 冻结BC之后的特权伺服和原生PhysX保守过滤；初始化接触/倾倒门禁与真实调用记账。
- 审计后的专家cam01原生视频重放；资产/权重/视频外置，源码保持轻量。

## 诚实边界

旧BC真实推理241次，初始化可行12/20，条件成功0/12；7倾倒、4碰撞、1失衡；工作流0/20。安全执行层不是纯BC，仍未通过plan24试点；v2 G0未通过，策略成功率null。专家25秒视频不是BC成功。PPO热启动与新场景研究尚未完成。

源公寓、纹理、厂商G1/Dex3、腿部ONNX和checkpoint不随源码提供。没有这些许可资产不能原样重现历史公寓；30分钟承诺仅针对资产无关离线快检，不包含首次SDK下载、完整45次扫参或机器人训练。

## 快速开始

按README创建Python3.12虚拟环境，安装requirements.txt和editable包，执行`python -m pipeline smoke`及`python -m pytest -q`。原生SDK已有的使用者按docs/RUNBOOK.md执行完整导出/装配/M1–M4/BC/渲染。

pipeline是数据与仿真层，training是学习/评测/安全层，configs是脱敏模板，results仅结论与汇总，docs收敛架构与方法。COCO源码从误伤的忽略规则中恢复；历史视觉BC、空场景课程和一次性失败恢复脚本不再作为首版入口。

## 打包与许可

源码ZIP和干净Git bundle在本地交付目录，校验和由外部SHA256SUMS.txt提供。仓库仅有新的首版根历史和v1.0.0本地tag；原私有实验仓完全保留。原创代码MIT，Unitree适配保留Apache-2.0；完整说明见NOTICE.md。

用户上传到新的空私有仓后，再手动打开Settings修改Public，或在Releases页面选择v1.0.0草拟发布。旧simforge远程有未清洗历史，不能直接公开；详见docs/PUBLISH_MANUALLY.md。

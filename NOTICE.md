# 许可边界与依赖

本仓MIT仅适用于本项目原创源码与文档；不替代第三方许可。

- Epic Unreal Engine及USD/Python Editor插件：按Epic条款在用户本机运行，不随仓分发。
- NVIDIA Isaac Sim4.5、Isaac Lab2.1.0及Omniverse/PhysX/Replicator：按其相应上游许可使用。库不随仓分发。
- Unitree G1/Dex3、厂商行走ONNX与第三方机器人配置：用户自行按上游条款取得。本仓没有模型或策略权重。G1是Unitree机器人，不能标为NVIDIA制造。
- `pipeline/ue_isaac_pipeline/unitree_wbc.py`的观测/动作适配改编自Unitree Robotics（Copyright 2025），该文件保留Apache-2.0许可，[许可全文](docs/licenses/Apache-2.0.txt)与[上游归属声明](docs/licenses/Unitree-Apache-2.0.txt)随仓保留。上游固定为`unitreerobotics/unitree_sim_isaaclab`的`e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc`提交，`action_provider/action_provider_wh_dds.py`。本项目MIT不覆盖这项Apache例外。
- 公寓、美术模型、纹理、视频中的第三方资产：不通过本仓重新许可或分发。外部展示授权不等于允许下载后再分发。
- NumPy、SciPy、Pillow、PyYAML、jsonschema等按各自许可安装；FFmpeg按所使用版本/构建的许可自行取得。
- tests/fixtures/ciede2000-sharma.txt为Sharma、Wu、Dalal提供的34组CIEDE2000数值测试对，用于验证本仓实现；来源与论文见https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/，不含第三方实现代码。

原始私有实验输出留在本机；本仓只含小型脱敏摘要，不含个人信息、访问令牌或本机路径。

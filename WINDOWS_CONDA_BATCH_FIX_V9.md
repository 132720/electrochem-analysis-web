# v9 Windows Conda 批处理提前退出修复

## 故障复现
v8 在两个批处理中使用：
`conda run -n electrochem_pro python --version >nul 2>&1`
但没有 `call`。当 `conda` 在 Windows 中解析到 conda.bat 时，从一个 .bat 调用另一个 .bat 会直接交出控制权，返回不到原脚本后续行；所以出现“Checking if ...”后马上回到提示符、没有报错的症状。

## 修复
两个脚本中都改为：
`call conda run -n electrochem_pro python --version >nul 2>&1`
以保证检查结束后继续执行安装/启动逻辑。

## 如何使用
1. 解压 v9 完整项目（或仅覆盖旧项目同名的两个 .bat 脚本）。
2. 在 Anaconda Prompt 中进入 electrochem_app 文件夹。
3. 执行 `setup_electrochem_pro_env.bat`，等待它自动完成创建和依赖安装。
4. 完成后执行 `start_electrochem_app.bat`。

## 验证范围
已对安装脚本及启动脚本检查所有 conda 调用均由 `call` 前缀执行；完整 EIS Python 回归测试继续通过。
当前环境无法直接执行 Windows 批处理或实际 Conda 安装，因此请在 Windows 机器运行验证。

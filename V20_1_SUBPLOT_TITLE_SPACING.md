# v20.1 — CV / GCD / EIS 右侧四图标题间距修复

根据用户截图，右侧的 CV · Full curves 标题几乎压到上排图框的上边线。

- 本版在 `screens/madap_dashboard.py` 增加 `SUBPLOT_TITLE_GAP_PX = 16`。
- 通过 Plotly 标题 annotation 的 `yanchor="bottom"` 与 `yshift=16`，将四个子图标题上移，使标题底部与对应图框顶部之间留出像素距离。
- 保持 Plotly 自己计算的四个子图标题坐标；EIS 有额外相位纵轴，其 subplot axis 编号与 CV/GCD 不同，因此不从 axis 编号猜测标题位置。
- 将主四图顶部边距增加到 118 px，为首排上移标题留空间。
- 保留 2×2 布局、全实线坐标框、无网格线、子图间距、科研图形和 CV/GCD/EIS 现有计算。
- 需要更大间距时，可以修改上述常量（例如改成20或24），无须调整每个子图。

所有四图采用同样设置：CV、GCD、EIS。新增自动化回归测试，检查标题独立间距和可调值。

项目运行：在 `electrochem_pro` 环境下，进入包含 `app.py` 的目录后执行
`python -m streamlit run app.py`。

尚未在用户 Windows 浏览器中对不同屏幕宽度实测。若仍有视觉重叠，请附上全宽屏幕截图和浏览器缩放比例。

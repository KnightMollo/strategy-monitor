# Strategy Monitor

公开策略监控仪表盘（Streamlit），是 Local Dashboard 的剥离版：
保留策略信号、行情指标与扫描，移除所有回测、个人交易与账户价值数据。

## 页面

- Overview: Hold'em / TMT 信号卡 + RRG 市场健康
- Hold'em: 完整信号状态机与决策树（含震荡模式：QQQ 20 日波动率 >25% 开 / <20% 关，期间 TQQQ 仅跟随 RSI(2) 腿，否则 SGOV；规则文件 `lib/holdem_chop.py` 与 Local Dashboard 逐字节一致）
- TMT: RSI(2) 主信号 + RRG Idle 轮动参考 + RRG 地图；熊市狙击腿（QQQ <= 200SMA 时 RSI(2) < 5 买 TQQQ，RSI(2) > 60 / 5 个交易日 / -10% 出场，`tmt-bear-sniper-v1`）
- GEX Filter: 38 标的 Gamma Exposure 扫描与 8 过滤条件
- SPX BWB: 当日结构建议（仅指引，不跟踪持仓）

## TMT 固定规则

- 版本 `tmt-fixed-2w-spy-v1`，与 Local Dashboard 的纯策略模块一致，不依赖个人持仓或相邻项目目录。
- Idle 锚定 2019-06-07，每 14 天指定周五的已完成收盘才评估；目标变化才计划下一实际开盘轮动，周中不发新的 idle 订单。
- breadth < 40% → SGOV；达标且上一周期 Improving、本周期 Leading → 按固定板块顺序选择；无突破 → SPY。
- 数据固定从 2019-06-03 开始，不能用滚动两年窗口重置日历或 EMA。周五休市、提前收盘和执行日期均按 XNYS 日历。
- Overview / TMT 明确显示评估收盘（美东）、锁定时间、原执行日、下一次评估及历史。过期计划只供参考，不每天顺延为新订单。
- RSI 的每日入场和条件出场不受 idle 日历限制。公开版不读取持仓，价格止损（牛市腿 -15% / 熊市腿 -10%）和持有上限（21 / 5 个实际交易日）不能替代持仓感知的本地判断。
- `.cache/tmt_idle_tmt-fixed-2w-spy-v1.json` 保存公共评估历史；刷新价格不删除历史。远端服务须重新安装依赖并部署本次代码，重启若不保留磁盘则历史会从本次同步重新开始。

## 明确剥离内容

本项目不读取、不展示：

- 交易流水 CSV
- 账户持仓与账户总资产
- 任何个人 P/L 统计

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render 部署

1. Push repo to GitHub
2. Render -> New Web Service -> Connect repo
3. Settings
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Python: 3.11+

## 安全说明

可安全分享给朋友查看策略逻辑与公开行情信号，不包含个人账户与交易隐私数据。
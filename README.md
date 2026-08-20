# Strategy Monitor

公开策略监控仪表盘（Streamlit），是 Local Dashboard 的剥离版：
保留策略信号、行情指标与扫描，移除所有回测、个人交易与账户价值数据。

## 页面

- Overview: Hold'em / TMT 信号卡 + RRG 市场健康
- Hold'em: 完整信号状态机与决策树
- TMT: RSI(2) 主信号 + RRG Idle 轮动参考 + RRG 地图
- GEX Filter: 38 标的 Gamma Exposure 扫描与 8 过滤条件
- SPX BWB: 当日结构建议（仅指引，不跟踪持仓）

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
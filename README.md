# Strategy Monitor

公开策略信号仪表盘 — 展示策略逻辑和实时信号，不包含任何个人交易数据。

## 包含策略

### Hold'em — 3x杠杆轮动
- RSI(10) 3-phase state machine
- Bull: TQQQ → UVXY hedge → SGOV cooldown
- Bear: SQQQ / TLT rotation

### TMT — RSI(2) 均值回归
- QQQ RSI(2) < 15 超卖 → 买入 TQQQ
- RSI(2) > 80 → 卖出
- 持仓上限 10 天

## 部署到 Render

1. Push this repo to GitHub
2. On Render → New Web Service → Connect repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Environment:** Python 3.11+

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 安全说明

此仓库**不包含**:
- 个人交易记录 (trades CSV)
- 账户持仓/市值数据
- 密码或 API keys

可安全分享给朋友查看策略逻辑和信号。
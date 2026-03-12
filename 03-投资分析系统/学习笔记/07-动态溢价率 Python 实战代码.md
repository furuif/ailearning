# 动态溢价率 - Python 实战代码

> 📅 创建日期：2026-03-12
> 🐍 语言：Python 3.9+
> 📊 目标：完整可运行的动态溢价率分析代码

---

## 一、环境准备

### 依赖安装

```bash
pip install akshare pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

### 导入模块

```python
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 二、数据获取模块

### 2.1 获取新股数据

```python
class IPODataFetcher:
    """IPO 数据获取类"""
    
    def __init__(self):
        pass
    
    def fetch_ipo_list(self, start_date='20230101', end_date='20260312'):
        """
        获取 IPO 列表
        
        参数:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
        
        返回:
            DataFrame: IPO 列表
        """
        try:
            # 获取新股上市首日行情
            ipo_df = ak.stock_new_gh_cninfo()
            
            # 数据清洗
            ipo_df = ipo_df[ipo_df['上市日期'] >= start_date]
            ipo_df = ipo_df[ipo_df['上市日期'] <= end_date]
            
            # 转换日期格式
            ipo_df['上市日期'] = pd.to_datetime(ipo_df['上市日期'])
            
            return ipo_df
        except Exception as e:
            print(f"获取 IPO 数据失败：{e}")
            return pd.DataFrame()
    
    def fetch_ipo_detail(self, stock_code):
        """
        获取单只新股详细信息
        
        参数:
            stock_code: 股票代码
        
        返回:
            dict: 新股详细信息
        """
        try:
            # 获取发行数据
            issue_df = ak.stock_new_ipo_cninfo()
            detail = issue_df[issue_df['股票代码'] == stock_code].iloc[0]
            
            return detail.to_dict()
        except Exception as e:
            print(f"获取新股详情失败：{e}")
            return {}
    
    def fetch_stock_history(self, stock_code, start_date, end_date):
        """
        获取个股历史行情
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            DataFrame: 历史行情
        """
        try:
            # 转换股票代码格式
            if stock_code.startswith('6'):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"
            
            # 获取历史行情
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"  # 前复权
            )
            
            return df
        except Exception as e:
            print(f"获取历史行情失败：{e}")
            return pd.DataFrame()
    
    def fetch_industry_pe(self, industry_name):
        """
        获取行业市盈率
        
        参数:
            industry_name: 行业名称
        
        返回:
            float: 行业平均 PE
        """
        try:
            # 获取行业估值数据
            industry_df = ak.stock_board_industry_cons_em(symbol=industry_name)
            avg_pe = industry_df['市盈率'].mean()
            return avg_pe
        except Exception as e:
            print(f"获取行业 PE 失败：{e}")
            return None
```

### 2.2 获取宏观数据

```python
class MacroDataFetcher:
    """宏观数据获取类"""
    
    def __init__(self):
        pass
    
    def fetch_pmi(self, start_date='20140101', end_date='20260312'):
        """获取 PMI 数据"""
        try:
            pmi_df = ak.macro_pmi_cn()
            pmi_df['时间'] = pd.to_datetime(pmi_df['时间'])
            return pmi_df
        except Exception as e:
            print(f"获取 PMI 数据失败：{e}")
            return pd.DataFrame()
    
    def fetch_social_financing(self, start_date='20140101', end_date='20260312'):
        """获取社融数据"""
        try:
            sf_df = ak.macro_social_finance_cn()
            sf_df['时间'] = pd.to_datetime(sf_df['时间'])
            return sf_df
        except Exception as e:
            print(f"获取社融数据失败：{e}")
            return pd.DataFrame()
    
    def fetch_money_supply(self, start_date='20140101', end_date='20260312'):
        """获取货币供应量数据（M1、M2）"""
        try:
            m_df = ak.macro_money_supply_cn()
            m_df['时间'] = pd.to_datetime(m_df['时间'])
            return m_df
        except Exception as e:
            print(f"获取货币供应量数据失败：{e}")
            return pd.DataFrame()
    
    def fetch_cpi_ppi(self, start_date='20140101', end_date='20260312'):
        """获取 CPI、PPI 数据"""
        try:
            cpi_df = ak.macro_cpi_cn()
            ppi_df = ak.macro_ppi_cn()
            
            cpi_df['时间'] = pd.to_datetime(cpi_df['时间'])
            ppi_df['时间'] = pd.to_datetime(ppi_df['时间'])
            
            return cpi_df, ppi_df
        except Exception as e:
            print(f"获取 CPI/PPI 数据失败：{e}")
            return pd.DataFrame(), pd.DataFrame()
```

---

## 三、动态溢价率计算模块

### 3.1 基础溢价率计算

```python
class PremiumCalculator:
    """溢价率计算器"""
    
    def __init__(self, issue_price, listing_date):
        """
        初始化
        
        参数:
            issue_price: 发行价
            listing_date: 上市日期
        """
        self.issue_price = issue_price
        self.listing_date = pd.to_datetime(listing_date)
    
    def base_premium(self, current_price):
        """
        计算基础溢价率
        
        参数:
            current_price: 当前价格
        
        返回:
            float: 基础溢价率 (%)
        """
        return (current_price - self.issue_price) / self.issue_price * 100
    
    def time_adjusted_premium(self, current_price, current_date):
        """
        计算时间衰减调整溢价率
        
        参数:
            current_price: 当前价格
            current_date: 当前日期
        
        返回:
            float: 调整后溢价率 (%)
        """
        days = (pd.to_datetime(current_date) - self.listing_date).days
        base = self.base_premium(current_price)
        
        # 时间衰减因子
        if days <= 5:
            decay_factor = 1.0
        elif days <= 20:
            decay_factor = np.exp(-0.05 * (days - 5))
        elif days <= 60:
            decay_factor = np.exp(-0.05 * 15) * np.exp(-0.02 * (days - 20))
        else:
            decay_factor = 0.3
        
        return base / decay_factor, days, decay_factor
    
    def market_adjusted_premium(self, current_price, index_start, index_current):
        """
        计算市场调整溢价率 (MAP)
        
        参数:
            current_price: 当前价格
            index_start: 上市日大盘指数
            index_current: 当前大盘指数
        
        返回:
            float: MAP (%)
        """
        stock_return = (current_price - self.issue_price) / self.issue_price
        market_return = (index_current - index_start) / index_start
        
        map_rate = (1 + stock_return) / (1 + market_return) - 1
        return map_rate * 100
    
    def premium_series(self, price_series, dates_series):
        """
        计算溢价率时间序列
        
        参数:
            price_series: 价格序列
            dates_series: 日期序列
        
        返回:
            DataFrame: 溢价率序列
        """
        results = []
        for price, date in zip(price_series, dates_series):
            base = self.base_premium(price)
            adjusted, days, decay = self.time_adjusted_premium(price, date)
            
            results.append({
                '日期': date,
                '价格': price,
                '天数': days,
                '基础溢价率': base,
                '时间调整溢价率': adjusted,
                '衰减因子': decay
            })
        
        return pd.DataFrame(results)
```

### 3.2 CDPI 综合指数计算

```python
class CDPICalculator:
    """CDPI 综合指数计算器"""
    
    def __init__(self):
        self.weights = {
            'pe': 0.25,
            'peg': 0.25,
            'comparable': 0.20,
            'liquidity': 0.15,
            'risk': 0.10,
            'sentiment': 0.05
        }
    
    def calculate_cdpi(self, data):
        """
        计算 CDPI 综合指数
        
        参数:
            data: dict，包含以下字段
                - current_price: 当前价
                - issue_price: 发行价
                - ipo_pe: 新股 PE
                - industry_pe: 行业 PE
                - ipo_growth: 新股增速
                - industry_growth: 行业增速
                - turnover_rate: 换手率
                - industry_turnover: 行业平均换手
                - volatility: 波动率
                - sentiment_score: 情绪得分 (0-100)
                - float_ratio: 流通比例
        
        返回:
            dict: CDPI 计算结果
        """
        # 1. PE 相对溢价
        pe_premium = (data['ipo_pe'] - data['industry_pe']) / data['industry_pe']
        
        # 2. PEG 相对溢价
        ipo_peg = data['ipo_pe'] / data['ipo_growth']
        industry_peg = data['industry_pe'] / data['industry_growth']
        peg_premium = (ipo_peg - industry_peg) / industry_peg
        
        # 3. 可比公司溢价（简化，假设等于行业）
        comparable_premium = pe_premium * 0.8
        
        # 4. 流动性调整
        liquidity_ratio = data['turnover_rate'] / data['industry_turnover']
        liquidity_adj = liquidity_ratio * (data['float_ratio'] / 0.8)
        liquidity_premium = liquidity_adj - 1
        
        # 5. 风险调整
        base_premium = (data['current_price'] - data['issue_price']) / data['issue_price']
        annualized_premium = base_premium * (252 / 20)  # 假设持有 20 日
        rap = (annualized_premium - 0.03) / (data['volatility'] * np.sqrt(252))
        risk_premium = np.clip(rap / 2, -1, 1)  # 标准化到 -1~1
        
        # 6. 情绪调整
        sentiment_premium = (data['sentiment_score'] - 50) / 100
        
        # 加权合成
        cdpi = (
            self.weights['pe'] * pe_premium +
            self.weights['peg'] * peg_premium +
            self.weights['comparable'] * comparable_premium +
            self.weights['liquidity'] * liquidity_premium +
            self.weights['risk'] * risk_premium +
            self.weights['sentiment'] * sentiment_premium
        )
        
        # 转换为 0-100 分
        cdpi_score = np.clip((cdpi + 0.5) * 100, 0, 100)
        
        return {
            'cdpi_score': cdpi_score,
            'cdpi_raw': cdpi,
            'components': {
                'pe_premium': pe_premium * 100,
                'peg_premium': peg_premium * 100,
                'comparable_premium': comparable_premium * 100,
                'liquidity_premium': liquidity_premium * 100,
                'risk_premium': risk_premium * 100,
                'sentiment_premium': sentiment_premium * 100
            },
            'signal': self.get_signal(cdpi_score)
        }
    
    def get_signal(self, cdpi_score):
        """获取交易信号"""
        if cdpi_score > 80:
            return "严重高估 - 强烈卖出"
        elif cdpi_score > 60:
            return "高估 - 卖出"
        elif cdpi_score > 40:
            return "合理 - 持有"
        elif cdpi_score > 20:
            return "低估 - 买入"
        else:
            return "严重低估 - 强烈买入"
    
    def cycle_adjusted_cdpi(self, cdpi_score, cycle_stage):
        """
        周期调整 CDPI
        
        参数:
            cdpi_score: CDPI 分数
            cycle_stage: 周期阶段 (复苏/过热/滞胀/衰退)
        
        返回:
            float: 调整后 CDPI
        """
        cycle_weights = {
            '复苏': 1.2,
            '过热': 0.8,
            '滞胀': 0.6,
            '衰退': 1.0
        }
        
        weight = cycle_weights.get(cycle_stage, 1.0)
        adjusted = cdpi_score * weight
        
        return np.clip(adjusted, 0, 100), self.get_signal(adjusted)
```

---

## 四、可视化模块

### 4.1 溢价率走势图

```python
class PremiumVisualizer:
    """溢价率可视化类"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8-darkgrid'
        plt.style.use(self.style)
    
    def plot_premium_series(self, df, title='新股溢价率走势'):
        """
        绘制溢价率时间序列
        
        参数:
            df: DataFrame，包含日期、基础溢价率、时间调整溢价率
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 基础溢价率
        ax.plot(df['日期'], df['基础溢价率'], label='基础溢价率', linewidth=2)
        
        # 时间调整溢价率
        ax.plot(df['日期'], df['时间调整溢价率'], label='时间调整溢价率', 
                linewidth=2, linestyle='--')
        
        # 合理区间
        ax.axhline(y=20, color='green', linestyle=':', alpha=0.5, label='低估线')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='合理上限')
        ax.axhline(y=80, color='red', linestyle=':', alpha=0.5, label='高估线')
        
        # 填充区域
        ax.fill_between(df['日期'], 20, 50, alpha=0.1, color='green', label='合理区间')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('溢价率 (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_cdpi_components(self, cdpi_result):
        """
        绘制 CDPI 成分图
        
        参数:
            cdpi_result: CDPI 计算结果 dict
        """
        components = cdpi_result['components']
        names = list(components.keys())
        values = list(components.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 柱状图
        colors = ['red' if v > 20 else 'green' if v < -20 else 'gray' for v in values]
        bars = ax.bar(names, values, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 参考线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=-20, color='green', linestyle='--', alpha=0.5)
        
        ax.set_ylabel('贡献 (%)', fontsize=12)
        ax.set_title(f'CDPI 成分分析 (总分：{cdpi_result["cdpi_score"]:.1f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_premium_distribution(self, df, column='基础溢价率'):
        """
        绘制溢价率分布直方图
        
        参数:
            df: DataFrame
            column: 列名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 直方图
        ax.hist(df[column], bins=30, edgecolor='black', alpha=0.7, density=True)
        
        # KDE 曲线
        from scipy import stats
        kde = stats.gaussian_kde(df[column])
        x_range = np.linspace(df[column].min(), df[column].max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax.set_xlabel('溢价率 (%)', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title(f'{column} 分布', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

---

## 五、完整实战示例

### 5.1 单只新股完整分析

```python
def analyze_single_ipo(stock_code):
    """
    单只新股完整分析
    
    参数:
        stock_code: 股票代码
    """
    print(f"=== 分析新股：{stock_code} ===\n")
    
    # 1. 数据获取
    fetcher = IPODataFetcher()
    ipo_detail = fetcher.fetch_ipo_detail(stock_code)
    
    if not ipo_detail:
        print("获取新股详情失败")
        return
    
    # 2. 提取关键数据
    issue_price = ipo_detail.get('发行价格', 0)
    listing_date = ipo_detail.get('上市日期', '')
    ipo_pe = ipo_detail.get('发行市盈率', 0)
    
    print(f"发行价：{issue_price} 元")
    print(f"上市日期：{listing_date}")
    print(f"发行 PE: {ipo_pe} 倍\n")
    
    # 3. 获取历史行情
    end_date = datetime.now()
    history = fetcher.fetch_stock_history(stock_code, listing_date, end_date)
    
    if history.empty:
        print("获取历史行情失败")
        return
    
    # 4. 计算溢价率
    calculator = PremiumCalculator(issue_price, listing_date)
    premium_df = calculator.premium_series(
        history['收盘'],
        history['日期']
    )
    
    # 5. 可视化
    viz = PremiumVisualizer()
    viz.plot_premium_series(premium_df, title=f'{stock_code} 溢价率走势')
    
    # 6. 计算最新 CDPI
    latest = premium_df.iloc[-1]
    cdpi_calc = CDPICalculator()
    
    cdpi_data = {
        'current_price': latest['价格'],
        'issue_price': issue_price,
        'ipo_pe': ipo_pe,
        'industry_pe': ipo_pe * 0.9,  # 假设行业 PE 为发行 PE 的 90%
        'ipo_growth': 30,  # 假设增速 30%
        'industry_growth': 20,
        'turnover_rate': 10,
        'industry_turnover': 5,
        'volatility': 5,
        'sentiment_score': 50,
        'float_ratio': 0.25
    }
    
    cdpi_result = cdpi_calc.calculate_cdpi(cdpi_data)
    print(f"\nCDPI 分析:")
    print(f"CDPI 分数：{cdpi_result['cdpi_score']:.1f}")
    print(f"交易信号：{cdpi_result['signal']}")
    
    return cdpi_result

# 运行示例
# analyze_single_ipo('688XXX')
```

### 5.2 批量分析多只新股

```python
def batch_analyze_ipos(stock_codes):
    """
    批量分析多只新股
    
    参数:
        stock_codes: 股票代码列表
    """
    results = []
    
    for code in stock_codes:
        try:
            result = analyze_single_ipo(code)
            if result:
                results.append({
                    '代码': code,
                    'CDPI': result['cdpi_score'],
                    '信号': result['signal']
                })
        except Exception as e:
            print(f"分析 {code} 失败：{e}")
            continue
    
    # 汇总结果
    summary_df = pd.DataFrame(results)
    print("\n=== 批量分析汇总 ===")
    print(summary_df.to_string(index=False))
    
    return summary_df

# 运行示例
# stock_codes = ['688XXX', '301XXX', '601XXX']
# batch_analyze_ipos(stock_codes)
```

---

## 六、回测框架

### 6.1 打新策略回测

```python
class IPOStrategyBacktester:
    """IPO 策略回测类"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.nav_history = [initial_capital]
    
    def backtest_subscription_strategy(self, ipo_list, strategy_params):
        """
        回测打新策略
        
        参数:
            ipo_list: IPO 列表 DataFrame
            strategy_params: 策略参数
                - min_cdpi: 最低 CDPI 分数
                - max_cdpi: 最高 CDPI 分数
                - holding_days: 持有天数
        """
        trades = []
        
        for idx, ipo in ipo_list.iterrows():
            # 计算 CDPI（简化）
            cdpi = self.calculate_ipo_cdpi(ipo)
            
            # 判断是否申购
            if strategy_params['min_cdpi'] <= cdpi <= strategy_params['max_cdpi']:
                # 申购
                shares = ipo['发行数量'] * 0.001  # 假设中签 0.1%
                cost = shares * ipo['发行价格']
                
                if self.capital >= cost:
                    self.capital -= cost
                    self.positions[ipo['股票代码']] = {
                        'shares': shares,
                        'cost': cost,
                        'listing_date': ipo['上市日期']
                    }
                    
                    trades.append({
                        '日期': ipo['上市日期'],
                        '代码': ipo['股票代码'],
                        '操作': '申购',
                        '数量': shares,
                        '价格': ipo['发行价格']
                    })
        
        return pd.DataFrame(trades)
    
    def calculate_ipo_cdpi(self, ipo):
        """简化 CDPI 计算"""
        # 实际应用中应调用 CDPICalculator
        pe_ratio = ipo.get('发行市盈率', 30) / 30  # 相对行业 PE
        growth = ipo.get('增速', 20) / 20
        
        cdpi = 50 + (1 - pe_ratio) * 20 + (growth - 1) * 10
        return np.clip(cdpi, 0, 100)
    
    def calculate_returns(self, trades_df):
        """计算策略收益"""
        if trades_df.empty:
            return {'总收益': 0, '年化收益': 0, '夏普比率': 0}
        
        # 简化计算
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        return {
            '总收益': f"{total_return:.2%}",
            '初始资金': self.initial_capital,
            '最终资金': self.capital
        }
```

---

## 七、实盘监控脚本

### 7.1 每日新股监控

```python
def daily_ipo_monitor():
    """
    每日新股监控
    
    功能:
    1. 获取今日上市新股
    2. 计算 CDPI
    3. 生成投资建议
    4. 发送提醒
    """
    print("=== 每日新股监控 ===")
    print(f"日期：{datetime.now().strftime('%Y-%m-%d')}\n")
    
    # 1. 获取今日上市新股
    fetcher = IPODataFetcher()
    today = datetime.now().strftime('%Y%m%d')
    
    # 简化：获取最近 7 日新股
    ipo_list = fetcher.fetch_ipo_list(
        start_date=(datetime.now() - timedelta(days=7)).strftime('%Y%m%d'),
        end_date=today
    )
    
    if ipo_list.empty:
        print("今日无新股上市")
        return
    
    print(f"今日新股数量：{len(ipo_list)}\n")
    
    # 2. 分析每只新股
    cdpi_calc = CDPICalculator()
    recommendations = []
    
    for idx, ipo in ipo_list.iterrows():
        code = ipo['股票代码']
        name = ipo['股票简称']
        
        # 简化 CDPI 计算
        cdpi_data = {
            'current_price': ipo['收盘价'],
            'issue_price': ipo['发行价'],
            'ipo_pe': ipo['发行市盈率'],
            'industry_pe': ipo['发行市盈率'] * 0.9,
            'ipo_growth': 25,
            'industry_growth': 20,
            'turnover_rate': 15,
            'industry_turnover': 5,
            'volatility': 5,
            'sentiment_score': 50,
            'float_ratio': 0.25
        }
        
        result = cdpi_calc.calculate_cdpi(cdpi_data)
        
        recommendations.append({
            '代码': code,
            '名称': name,
            'CDPI': f"{result['cdpi_score']:.1f}",
            '信号': result['signal']
        })
    
    # 3. 输出结果
    rec_df = pd.DataFrame(recommendations)
    print(rec_df.to_string(index=False))
    
    # 4. 保存结果
    output_file = f"新股监控_{today}.csv"
    rec_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至：{output_file}")

# 运行示例
# daily_ipo_monitor()
```

---

## 八、代码使用指南

### 8.1 快速开始

```python
# 1. 导入模块
from premium_analyzer import PremiumCalculator, CDPICalculator, PremiumVisualizer

# 2. 创建计算器
calc = PremiumCalculator(issue_price=50, listing_date='2026-02-20')

# 3. 计算溢价率
base = calc.base_premium(80)
adjusted, days, decay = calc.time_adjusted_premium(80, '2026-03-12')

print(f"基础溢价率：{base:.1f}%")
print(f"调整后溢价率：{adjusted:.1f}%")
print(f"上市天数：{days}")

# 4. 计算 CDPI
cdpi = CDPICalculator()
result = cdpi.calculate_cdpi({...})
print(f"CDPI: {result['cdpi_score']:.1f}")
print(f"信号：{result['signal']}")
```

### 8.2 注意事项

1. **数据源**：AKShare 部分接口可能不稳定，建议添加重试机制
2. **实时性**：CDPI 计算需要最新数据，建议每日更新
3. **参数校准**：权重和阈值需根据历史数据回测优化
4. **风险控制**：代码仅供参考，实盘需谨慎

---

*本笔记由 AI 助手整理生成*

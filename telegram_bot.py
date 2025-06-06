import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, JobQueue
from signal_generator import generate_signal, get_cached_indicators, backtest_strategy
from config import BOT_TOKEN
from logger import log_trade, get_trade_logs
from datetime import datetime
import matplotlib.pyplot as plt
import io
import numpy as np
from fundamental_analysis import scrape_economic_calendar

# Simpan preferensi user dan simulasi
user_prefs = {}
user_simulations = {}
last_econ_notification = {}
last_ews_notification = {}

# Menu
def main_menu(chat_id):
    keyboard = [
        [telegram.InlineKeyboardButton("📈 Get Signal", callback_data=f"get_signal_{chat_id}"),
         telegram.InlineKeyboardButton("⚙️ Settings", callback_data=f"settings_menu_{chat_id}")],
        [telegram.InlineKeyboardButton("📊 Dashboard", callback_data=f"dashboard_{chat_id}"),
         telegram.InlineKeyboardButton("📉 Simulate Trade", callback_data=f"simulate_trade_{chat_id}")],
        [telegram.InlineKeyboardButton("📊 Visualize Signal", callback_data=f"visualize_signal_{chat_id}"),
         telegram.InlineKeyboardButton("🔄 Backtest", callback_data=f"backtest_{chat_id}")],
        [telegram.InlineKeyboardButton("ℹ️ About", callback_data=f"about_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

def settings_menu(chat_id):
    keyboard = [
        [telegram.InlineKeyboardButton("⏰ Timeframe", callback_data=f"timeframe_menu_{chat_id}"),
         telegram.InlineKeyboardButton("🎯 Strategy", callback_data=f"strategy_menu_{chat_id}")],
        [telegram.InlineKeyboardButton("⬅️ Back", callback_data=f"back_to_main_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

def timeframe_menu(chat_id):
    keyboard = [
        [telegram.InlineKeyboardButton("5m", callback_data=f"timeframe_5_{chat_id}"),
         telegram.InlineKeyboardButton("15m", callback_data=f"timeframe_15_{chat_id}"),
         telegram.InlineKeyboardButton("1H", callback_data=f"timeframe_1H_{chat_id}")],
        [telegram.InlineKeyboardButton("⬅️ Back", callback_data=f"settings_menu_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

def strategy_menu(chat_id):
    keyboard = [
        [telegram.InlineKeyboardButton("Scalping", callback_data=f"strategy_scalping_{chat_id}"),
         telegram.InlineKeyboardButton("Intraday", callback_data=f"strategy_intraday_{chat_id}"),
         telegram.InlineKeyboardButton("Swing", callback_data=f"strategy_swing_{chat_id}")],
        [telegram.InlineKeyboardButton("Mean Reversion", callback_data=f"strategy_mean_reversion_{chat_id}"),
         telegram.InlineKeyboardButton("⬅️ Back", callback_data=f"settings_menu_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

def signal_menu(chat_id, timeframe, strategy, signal_id):
    keyboard = [
        [telegram.InlineKeyboardButton("🔍 Technical Details", callback_data=f"tech_details_{chat_id}"),
         telegram.InlineKeyboardButton("📰 Fundamental Details", callback_data=f"fund_details_{chat_id}")],
        [telegram.InlineKeyboardButton("📌 Risk Analysis", callback_data=f"risk_details_{chat_id}"),
         telegram.InlineKeyboardButton("🔄 Refresh Signal", callback_data=f"get_signal_{chat_id}")],
        [telegram.InlineKeyboardButton(f"⏰ Change TF ({timeframe})", callback_data=f"timeframe_menu_{chat_id}"),
         telegram.InlineKeyboardButton(f"🎯 Change Strategy ({strategy})", callback_data=f"strategy_menu_{chat_id}")],
        [telegram.InlineKeyboardButton("📊 Visualize Signal", callback_data=f"visualize_signal_{chat_id}"),
         telegram.InlineKeyboardButton("👍 Feedback", callback_data=f"feedback_{signal_id}_{chat_id}")],
        [telegram.InlineKeyboardButton("⬅️ Main Menu", callback_data=f"back_to_main_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

# Fungsi Utama
def send_signal(context, chat_id, auto=False):
    timeframe = user_prefs.get(chat_id, {}).get('timeframe', '15')
    strategy = user_prefs.get(chat_id, {}).get('strategy', 'intraday')
    signal = generate_signal(chat_id, timeframe, strategy)
    if not signal:
        context.bot.send_message(chat_id=chat_id, text="⚠️ *No high-quality signal available right now.*\nWaiting for confluence...", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return
    
    signal_id = str(int(datetime.now().timestamp()))
    signal['signal_id'] = signal_id
    message = (
        f"🚨 *XAUUSD SIGNAL* 🚨\n\n"
        f"📅 *{datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}*\n"
        f"📈 *Price:* {signal['price']:.2f}\n"
        f"🔮 *Predicted:* {signal['predicted_price']:.2f}\n"
        f"⭐ *Confidence:* {signal['confidence']} ({signal['confidence_score']:.0f}%)\n"
        f"📊 *Bias:* {signal['bias']}\n"
        f"💡 *Action:* _{signal['recommendation']}_\n"
        f"🛑 *SL:* {signal['stop_loss']:.2f}\n"
        f"🎯 *TP1:* {signal['take_profit_1']:.2f} (R/R: {signal['rr_ratio_1']:.2f})\n"
        f"🎯 *TP2:* {signal['take_profit_2']:.2f} (R/R: {signal['rr_ratio_2']:.2f})\n"
        f"🎯 *TP3:* {signal['take_profit_3']:.2f} (R/R: {signal['rr_ratio_3']:.2f})\n"
        f"📈 *R/R Avg:* {signal['rr_ratio']:.2f}\n"
        f"💰 *Position Size:* {signal['position_size']:.2f}\n"
        f"⏰ *TF:* {signal['timeframe']}\n"
        f"🎯 *Strategy:* {signal['strategy'].capitalize()}\n"
        f"📉 *Market Regime:* {signal['market_regime']}\n"
        f"🛡️ *Trailing Stop:* {signal['atr_trailing_stop']:.2f}\n\n"
        f"🔍 *Click buttons below for more details!*"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=signal_menu(chat_id, timeframe, strategy, signal_id))
    log_trade(chat_id, signal)

def send_tech_details(context, chat_id):
    signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'))
    if not signal:
        return
    fib_info = '\n'.join([f"  • {key}: {value:.2f}" for key, value in signal['fib_levels'].items()])
    pivot_info = f"  • Pivot: {signal['pivot_points']['pivot']:.2f}\n  • S1: {signal['pivot_points']['s1']:.2f}\n  • R1: {signal['pivot_points']['r1']:.2f}"
    message = (
        f"🔍 *Technical Analysis Details* 🔍\n\n"
        f"📊 *RSI:* {signal['rsi']:.2f}\n"
        f"📈 *MACD Hist:* {signal['macd_histogram']:.2f}\n"
        f"📉 *Stochastic:* %K={signal['stoch_k']:.2f}, %D={signal['stoch_d']:.2f}\n"
        f"📊 *ADX:* {signal['adx']:.2f}\n"
        f"📉 *CCI:* {signal['cci']:.2f}\n"
        f"📈 *SAR:* {signal['sar']:.2f}\n"
        f"📉 *VWAP:* {signal['vwap']:.2f}\n"
        f"⚡ *Volatility:* {signal['volatility']:.2f}\n"
        f"📉 *Fibonacci Levels:*\n{fib_info}\n"
        f"📈 *Pivot Points:*\n{pivot_info}\n"
        f"🕯️ *Candlestick:* {signal['candlestick_pattern']}\n"
        f"🌊 *Elliott Wave:* {signal['elliott_wave']}\n"
        f"📐 *Harmonic:* {signal['harmonic_pattern']}\n"
        f"📏 *Gann Angle:* {signal['gann_angles']['1x1']:.2f}\n"
        f"⏳ *Cycle Period:* {signal['cycle_period']:.2f}\n"
        f"🔳 *Fractal:* {signal['fractal']}\n"
        f"📊 *VSA:* {signal['vsa_signal']}\n"
        f"📉 *Market Delta:* {signal['market_delta']:.2f}\n"
        f"📊 *Market Profile:* VA Low={signal['va_low']:.2f}, VA High={signal['va_high']:.2f}\n"
        f"📉 *Order Flow:* {signal['order_flow']:.2f}\n"
        f"📅 *Seasonality:* {signal['seasonality']:.2f}\n"
        f"🌐 *Wavelet Trend:* {signal['wavelet_trend']:.2f}\n"
        f"📈 *MMO:* {signal['mmo']:.2f}\n"
        f"📍 *PAZ:* Supply={signal['paz']['supply']:.2f}, Demand={signal['paz']['demand']:.2f}\n"
        f"🛡️ *ATR Trailing Stop:* {signal['atr_trailing_stop']:.2f}\n"
        f"📉 *Renko Trend:* {signal['renko_trend']}\n"
        f"⚡ *DVI:* {signal['dvi']:.2f}\n"
        f"📈 *Keltner Channels:* Upper={signal['keltner_upper']:.2f}, Lower={signal['keltner_lower']:.2f}\n"
        f"📊 *Elder Ray:* Bull={signal['elder_ray_bull']:.2f}, Bear={signal['elder_ray_bear']:.2f}\n"
        f"📉 *CVD:* {signal['cvd']:.2f}\n"
        f"📈 *Trend:* {signal['trend']:.2f}\n"
        f"📅 *Seasonal:* {signal['seasonal']:.2f}\n"
        f"📉 *Residual:* {signal['residual']:.2f}\n"
        f"🌐 *Market Breadth:* {signal['market_breadth']:.2f}\n"
        f"📈 *VWM:* {signal['vwm']:.2f}\n"
        f"⚡ *Volatility Breakout:* {signal['volatility_breakout']}"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=signal_menu(chat_id, signal['timeframe'], signal['strategy'], signal.get('signal_id', '0')))

def send_fund_details(context, chat_id):
    signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'))
    if not signal:
        return
    econ_info = '\n'.join([f"  • {e['event']} ({e['impact']}) at {e['time']}" for e in signal['econ_events']]) if signal['econ_events'] else "  • None"
    message = (
        f"📰 *Fundamental Analysis Details* 📰\n\n"
        f"📊 *News Sentiment:* {signal['news_sentiment']:.2f}\n"
        f"📈 *X Sentiment:* {signal['x_sentiment']:.2f}\n"
        f"🌡️ *Audio Sentiment:* {signal['audio_sentiment']:.2f}\n"
        f"🌡️ *Sentiment Heatmap:* {signal['sentiment_heatmap']:.2f}\n"
        f"📅 *Economic Events:*\n{econ_info}\n"
        f"📈 *COT Position:* {signal['cot_position']}\n"
        f"📉 *VIX Sentiment:* {signal['vix_sentiment']}\n"
        f"🔗 *Correlation:*\n  • EURUSD: {signal['correlation'].get('EURUSD', 0):.2f}\n  • USDJPY: {signal['correlation'].get('USDJPY', 0):.2f}\n  • DXY: {signal['correlation'].get('DXY', 0):.2f}\n"
        f"💸 *Hedging:* {signal['hedging']}"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=signal_menu(chat_id, signal['timeframe'], signal['strategy'], signal.get('signal_id', '0')))

def send_risk_details(context, chat_id):
    signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'))
    if not signal:
        return
    message = (
        f"📌 *Risk Analysis Details* 📌\n\n"
        f"📈 *Expected Return (Monte Carlo):* {signal['expected_return']:.2f}\n"
        f"📉 *Risk Std Dev:* {signal['risk_std']:.2f}\n"
        f"📊 *Strategy:* {signal['strategy'].upper()}\n"
        f"📉 *Market Pattern Cluster:* {signal['cluster']}\n"
        f"⚠️ *Anomaly Detected:* {'Yes' if signal['anomaly'] else 'No'}\n"
        f"💰 *Position Size:* {signal['position_size']:.2f}"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=signal_menu(chat_id, signal['timeframe'], signal['strategy'], signal.get('signal_id', '0')))

def send_dashboard(context, chat_id):
    trades = get_trade_logs(chat_id)
    if not trades:
        context.bot.send_message(chat_id=chat_id, text="📊 *No trades recorded yet!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return
    
    total_trades = len(trades)
    wins = sum(1 for trade in trades if (trade['signal']['recommendation'] == 'BUY' and trade['signal']['price'] < trade['signal']['take_profit_1']) or
               (trade['signal']['recommendation'] == 'SELL' and trade['signal']['price'] > trade['signal']['take_profit_1']))
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_pips = sum((trade['signal']['take_profit_1'] - trade['signal']['price']) if trade['signal']['recommendation'] == 'BUY' else
                     (trade['signal']['price'] - trade['signal']['take_profit_1']) for trade in trades)
    avg_confidence = sum(trade['signal']['confidence_score'] for trade in trades) / total_trades if total_trades > 0 else 0

    message = (
        f"📊 *Trading Dashboard* 📊\n\n"
        f"📅 *As of {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}*\n"
        f"📈 *Total Trades:* {total_trades}\n"
        f"🏆 *Win Rate:* {win_rate:.2f}%\n"
        f"💰 *Total Pips:* {total_pips:.2f}\n"
        f"⭐ *Avg Confidence:* {avg_confidence:.2f}%\n"
        f"📉 *Recent Trade:*\n"
        f"  • Action: {trades[-1]['signal']['recommendation']}\n"
        f"  • Price: {trades[-1]['signal']['price']:.2f}\n"
        f"  • Result: {'Win' if (trades[-1]['signal']['recommendation'] == 'BUY' and trades[-1]['signal']['price'] < trades[-1]['signal']['take_profit_1']) else 'Loss'}\n"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=main_menu(chat_id))

def visualize_signal(context, chat_id):
    signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'))
    if not signal:
        context.bot.send_message(chat_id=chat_id, text="⚠️ *No signal to visualize!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return

    from data_fetcher import historical_data
    from config import CURRENCY_PAIR
    prices = historical_data[CURRENCY_PAIR][-50:]
    timestamps = np.arange(len(prices))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, prices, label='Price', color='blue')
    plt.axhline(signal['price'], color='black', linestyle='--', label=f'Current Price ({signal["price"]:.2f})')
    plt.axhline(signal['stop_loss'], color='red', linestyle='--', label=f'SL ({signal["stop_loss"]:.2f})')
    plt.axhline(signal['take_profit_1'], color='green', linestyle='--', label=f'TP1 ({signal["take_profit_1"]:.2f})')
    plt.axhline(signal['take_profit_2'], color='green', linestyle='--', label=f'TP2 ({signal["take_profit_2"]:.2f})')
    plt.axhline(signal['take_profit_3'], color='green', linestyle='--', label=f'TP3 ({signal["take_profit_3"]:.2f})')
    for level, value in signal['fib_levels'].items():
        plt.axhline(value, color='purple', linestyle=':', label=f'Fib {level} ({value:.2f})')
    plt.title('XAUUSD Signal Visualization')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    context.bot.send_photo(chat_id=chat_id, photo=buf, caption="📊 *Signal Visualization*", parse_mode='Markdown', reply_markup=signal_menu(chat_id, signal['timeframe'], signal['strategy'], signal.get('signal_id', '0')))

def simulate_trade(context, chat_id):
    if chat_id not in user_simulations:
        user_simulations[chat_id] = {'balance': 10000, 'trades': []}

    from data_fetcher import historical_data
    from config import CURRENCY_PAIR
    if len(historical_data[CURRENCY_PAIR]) < 100:
        context.bot.send_message(chat_id=chat_id, text="⚠️ *Not enough historical data for simulation!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return

    sim_data = historical_data[CURRENCY_PAIR][-100:-50]
    sim_signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'), sim_data=sim_data)
    if not sim_signal:
        context.bot.send_message(chat_id=chat_id, text="⚠️ *No signal for simulation!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return

    future_prices = historical_data[CURRENCY_PAIR][-50:]
    entry_price = sim_signal['price']
    stop_loss = sim_signal['stop_loss']
    take_profit_1 = sim_signal['take_profit_1']
    take_profit_2 = sim_signal['take_profit_2']
    take_profit_3 = sim_signal['take_profit_3']
    position_size = sim_signal['position_size']

    result = 0
    for price in future_prices:
        if sim_signal['recommendation'] == 'BUY':
            if price >= take_profit_3:
                result = (take_profit_3 - entry_price) * position_size
                break
            elif price >= take_profit_2:
                result = (take_profit_2 - entry_price) * position_size * 0.5
                break
            elif price >= take_profit_1:
                result = (take_profit_1 - entry_price) * position_size * 0.3
                break
            elif price <= stop_loss:
                result = (stop_loss - entry_price) * position_size
                break
        else:  # SELL
            if price <= take_profit_3:
                result = (entry_price - take_profit_3) * position_size
                break
            elif price <= take_profit_2:
                result = (entry_price - take_profit_2) * position_size * 0.5
                break
            elif price <= take_profit_1:
                result = (entry_price - take_profit_1) * position_size * 0.3
                break
            elif price >= stop_loss:
                result = (entry_price - stop_loss) * position_size
                break

    if result == 0:
        result = (future_prices[-1] - entry_price) * position_size if sim_signal['recommendation'] == 'BUY' else (entry_price - future_prices[-1]) * position_size

    user_simulations[chat_id]['balance'] += result
    user_simulations[chat_id]['trades'].append({
        'signal': sim_signal,
        'result': result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')
    })

    message = (
        f"📉 *Trade Simulation Result* 📉\n\n"
        f"📅 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}*\n"
        f"📈 *Action:* {sim_signal['recommendation']}\n"
        f"💰 *Entry Price:* {entry_price:.2f}\n"
        f"🛑 *SL:* {stop_loss:.2f}\n"
        f"🎯 *TP1:* {take_profit_1:.2f}\n"
        f"🎯 *TP2:* {take_profit_2:.2f}\n"
        f"🎯 *TP3:* {take_profit_3:.2f}\n"
        f"📊 *Result:* {result:.2f} USD\n"
        f"💵 *New Balance:* {user_simulations[chat_id]['balance']:.2f} USD\n"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=main_menu(chat_id))

def send_backtest(context, chat_id):
    timeframe = user_prefs.get(chat_id, {}).get('timeframe', '15')
    strategy = user_prefs.get(chat_id, {}).get('strategy', 'intraday')
    from data_fetcher import historical_data
    from config import CURRENCY_PAIR
    if len(historical_data[CURRENCY_PAIR]) < 100:
        context.bot.send_message(chat_id=chat_id, text="⚠️ *Not enough historical data for backtesting!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        return

    backtest_result = backtest_strategy(historical_data[CURRENCY_PAIR][-500:], timeframe, strategy)
    message = (
        f"🔄 *Backtest Result* 🔄\n\n"
        f"📅 *Period:* Last 500 candles\n"
        f"⏰ *Timeframe:* {timeframe}m\n"
        f"🎯 *Strategy:* {strategy.capitalize()}\n"
        f"📈 *Win Rate:* {backtest_result['win_rate']:.2f}%\n"
        f"💰 *Total Profit:* {backtest_result['total_profit']:.2f} pips\n"
        f"📉 *Max Drawdown:* {backtest_result['max_drawdown']:.2f}%\n"
        f"⭐ *Avg Confidence:* {backtest_result['avg_confidence']:.2f}%\n"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=main_menu(chat_id))

def feedback_menu(signal_id, chat_id):
    keyboard = [
        [telegram.InlineKeyboardButton("✅ Win", callback_data=f"feedback_win_{signal_id}_{chat_id}"),
         telegram.InlineKeyboardButton("❌ Loss", callback_data=f"feedback_loss_{signal_id}_{chat_id}")],
        [telegram.InlineKeyboardButton("⬅️ Back", callback_data=f"back_to_main_{chat_id}")]
    ]
    return telegram.InlineKeyboardMarkup(keyboard)

def check_economic_events_and_ews(context):
    from datetime import timedelta
    econ_events = scrape_economic_calendar()
    current_time = datetime.now()
    for chat_id in user_prefs.keys():
        last_notified = last_econ_notification.get(chat_id, current_time - timedelta(hours=1))
        last_ews_notified = last_ews_notification.get(chat_id, current_time - timedelta(minutes=15))
        if (current_time - last_notified).total_seconds() < 3600:
            continue

        # Notifikasi event ekonomi
        for event in econ_events:
            if 'FOMC' in event['event'] or 'NFP' in event['event']:
                message = (
                    f"🚨 *High-Impact Event Alert!* 🚨\n\n"
                    f"📅 *Event:* {event['event']}\n"
                    f"⏰ *Time:* {event['time']}\n"
                    f"⚠️ *Impact:* {event['impact']}\n"
                    f"💡 *Advice:* Avoid trading during this event due to high volatility."
                )
                context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                last_econ_notification[chat_id] = current_time

        # Early Warning System
        if (current_time - last_ews_notified).total_seconds() < 900:
            continue

        signal = generate_signal(chat_id, user_prefs.get(chat_id, {}).get('timeframe', '15'), user_prefs.get(chat_id, {}).get('strategy', 'intraday'))
        if signal:
            if signal['volatility'] > 50:
                message = (
                    f"⚡ *High Volatility Alert!* ⚡\n\n"
                    f"📈 *Current Volatility:* {signal['volatility']:.2f}\n"
                    f"💡 *Advice:* Tighten stops or reduce position size."
                )
                context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                last_ews_notification[chat_id] = current_time

            if signal['anomaly']:
                message = (
                    f"🚨 *Market Anomaly Detected!* 🚨\n\n"
                    f"📉 *Details:* Unusual market behavior detected.\n"
                    f"💡 *Advice:* Avoid trading until conditions stabilize."
                )
                context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                last_ews_notification[chat_id] = current_time

            if signal['sentiment_heatmap'] > 0.8 or signal['sentiment_heatmap'] < -0.8:
                message = (
                    f"🌡️ *Extreme Sentiment Alert!* 🌡️\n\n"
                    f"📊 *Sentiment Heatmap:* {signal['sentiment_heatmap']:.2f}\n"
                    f"💡 *Advice:* Be cautious of potential reversals."
                )
                context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                last_ews_notification[chat_id] = current_time

def button(update, context):
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data.split('_')
    query.answer()

    if data[0] == 'get' and data[1] == 'signal':
        send_signal(context, chat_id)
    elif data[0] == 'dashboard':
        send_dashboard(context, chat_id)
    elif data[0] == 'visualize' and data[1] == 'signal':
        visualize_signal(context, chat_id)
    elif data[0] == 'simulate':
        simulate_trade(context, chat_id)
    elif data[0] == 'backtest':
        send_backtest(context, chat_id)
    elif data[0] == 'feedback':
        if len(data) == 2:
            signal_id = data[1]
            query.edit_message_text(text="📊 *Please provide feedback on this signal:*", parse_mode='Markdown', reply_markup=feedback_menu(signal_id, chat_id))
        elif data[1] == 'win':
            signal_id = data[2]
            log_trade(chat_id, {'signal_id': signal_id, 'feedback': 'win'})
            query.edit_message_text(text="✅ *Thank you for your feedback! Signal marked as WIN.*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
        elif data[1] == 'loss':
            signal_id = data[2]
            log_trade(chat_id, {'signal_id': signal_id, 'feedback': 'loss'})
            query.edit_message_text(text="❌ *Thank you for your feedback! Signal marked as LOSS.*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
    elif data[0] == 'about':
        message = (
            "ℹ️ *About This Bot*\n\n"
            "🔴 *Developer:* xAI Team\n"
            "📅 *Version:* 2.3.1 (Updated June 07, 2025)\n"
            "📈 *Purpose:* Provide high-quality XAUUSD trading signals using advanced AI technologies.\n"
            "⚙️ *Features:*\n"
            "  • Quantum-inspired optimization for risk/reward\n"
            "  • GNN-based market prediction\n"
            "  • Multimodal sentiment analysis (text + audio)\n"
            "  • Adaptive trailing stop with Deep Q-Learning\n"
            "  • Market regime detection with HMM\n"
            "  • Multiple TP levels (TP1, TP2, TP3)\n"
            "  • Backtesting and trading simulation\n"
            "📞 *Support:* Contact @GrokSupport on Telegram\n"
        )
        query.edit_message_text(text=message, parse_mode='Markdown', reply_markup=main_menu(chat_id))
    elif data[0] == 'settings' and data[1] == 'menu':
        query.edit_message_text(text="⚙️ *Settings Menu*", parse_mode='Markdown', reply_markup=settings_menu(chat_id))
    elif data[0] == 'timeframe' and data[1] == 'menu':
        query.edit_message_text(text="⏰ *Select Timeframe*", parse_mode='Markdown', reply_markup=timeframe_menu(chat_id))
    elif data[0] == 'strategy' and data[1] == 'menu':
        query.edit_message_text(text="🎯 *Select Strategy*", parse_mode='Markdown', reply_markup=strategy_menu(chat_id))
    elif data[0] == 'back' and data[1] == 'to' and data[2] == 'main':
        query.edit_message_text(text="👋 *Welcome to XAUUSD Trading Bot!*", parse_mode='Markdown', reply_markup=main_menu(chat_id))
    elif data[0] == 'tech' and data[1] == 'details':
        send_tech_details(context, chat_id)
    elif data[0] == 'fund' and data[1] == 'details':
        send_fund_details(context, chat_id)
    elif data[0] == 'risk' and data[1] == 'details':
        send_risk_details(context, chat_id)
    elif data[0] == 'timeframe':
        timeframe = data[1]
        if chat_id not in user_prefs:
            user_prefs[chat_id] = {}
        user_prefs[chat_id]['timeframe'] = timeframe
        query.edit_message_text(text=f"⏰ *Timeframe set to {timeframe}m!*", parse_mode='Markdown', reply_markup=settings_menu(chat_id))
    elif data[0] == 'strategy':
        strategy = data[1].lower()
        if chat_id not in user_prefs:
            user_prefs[chat_id] = {}
        user_prefs[chat_id]['strategy'] = strategy
        query.edit_message_text(text=f"🎯 *Strategy set to {strategy.capitalize()}!*", parse_mode='Markdown', reply_markup=strategy_menu(chat_id))

def start(update, context):
    chat_id = update.message.chat_id
    message = (
        "👋 *Welcome to XAUUSD Trading Bot!*\n\n"
        "I’m here to help you trade XAUUSD with cutting-edge AI technologies for maximum accuracy and profitability.\n\n"
        "🔍 *What I Offer:*\n"
        "  • Quantum-inspired risk/reward optimization\n"
        "  • GNN-based market predictions\n"
        "  • Multimodal sentiment analysis (text + audio)\n"
        "  • Adaptive trailing stop with Deep Q-Learning\n"
        "  • Market regime detection with HMM\n"
        "  • Multiple TP levels (TP1, TP2, TP3)\n"
        "  • Backtesting, visualization, and simulation\n\n"
        "📈 *Get started by selecting an option below!*"
    )
    context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=main_menu(chat_id))

def auto_signal(context):
    for chat_id in user_prefs.keys():
        send_signal(context, chat_id, auto=True)

def start_telegram_bot():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button))

    updater.job_queue.run_repeating(auto_signal, interval=900)  # 15 menit
    updater.job_queue.run_repeating(check_economic_events_and_ews, interval=300)  # 5 menit

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    start_telegram_bot()

"""
Trade risk management: position sizing, stop loss, take profit, circuit breakers.

Configuration constants at the top of the file set all defaults.
All pct values are fractions (0.02 = 2%), not percentages.

RiskManager.evaluate_trade() is the single entry point for any trade decision.
It returns a dict that downstream code uses to size and execute (or skip) trades.
"""
import logging

logger = logging.getLogger(__name__)

# ── Default risk parameters ────────────────────────────────────────────────────
DEFAULT_RISK_PER_TRADE = 0.02     # Fraction of capital risked per trade (2%)
DEFAULT_STOP_LOSS_PCT = 0.02      # Stop loss below entry price (2%)
DEFAULT_TAKE_PROFIT_PCT = 0.04    # Take profit above entry price (4%, 1:2 risk/reward)
DEFAULT_MAX_DRAWDOWN_PCT = 0.15   # Global circuit breaker: halt new trades at 15% drawdown
DEFAULT_MAX_DAILY_TRADES = 3      # Max new positions opened per calendar day
# ──────────────────────────────────────────────────────────────────────────────


class RiskManager:
    def __init__(
        self,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
        max_daily_trades: int = DEFAULT_MAX_DAILY_TRADES,
    ):
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_trades = max_daily_trades

    def evaluate_trade(
        self,
        signal: int,
        current_price: float,
        current_capital: float,
        peak_capital: float,
        daily_trades: int,
    ) -> dict:
        """
        Decides whether a trade can be executed and computes trade parameters.

        Args:
            signal: 1 = buy, -1 = sell/short, 0 = no position.
            current_price: Latest market price of the asset.
            current_capital: Current account equity.
            peak_capital: Highest equity reached so far (for drawdown calc).
            daily_trades: Number of trades already opened today.

        Returns:
            {
                "can_trade": bool,
                "position_size": float,  # number of shares/units to buy
                "stop_loss": float,      # price level for stop loss
                "take_profit": float,    # price level for take profit
                "reason": str,           # human-readable decision rationale
            }
        """
        _no_trade = dict(can_trade=False, position_size=0.0, stop_loss=0.0, take_profit=0.0)

        if signal == 0:
            return {**_no_trade, 'reason': 'neutral signal — no position'}

        drawdown = (peak_capital - current_capital) / peak_capital
        if drawdown >= self.max_drawdown_pct:
            msg = (
                f"max drawdown circuit breaker triggered: "
                f"{drawdown:.2%} >= {self.max_drawdown_pct:.2%}"
            )
            logger.critical("RISK HALT — %s", msg)
            return {**_no_trade, 'reason': msg}

        if daily_trades >= self.max_daily_trades:
            msg = f"daily trade limit reached ({daily_trades}/{self.max_daily_trades})"
            return {**_no_trade, 'reason': msg}

        # Fixed-fractional position sizing: risk RISK_PER_TRADE * capital per trade.
        dollar_risk = current_capital * self.risk_per_trade
        position_size = dollar_risk / current_price

        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct)

        return {
            'can_trade': True,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': (
                f"approved: size={position_size:.4f} units, "
                f"SL={stop_loss:.2f}, TP={take_profit:.2f}"
            ),
        }

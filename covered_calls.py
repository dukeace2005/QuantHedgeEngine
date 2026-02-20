"""
Institutional Covered Call Management & Rolling Engine v2.0
Production-Grade Implementation

Author: Quantitative Options Strategist
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class Signal(Enum):
    """Trading signals"""
    KEEP = "KEEP"
    ROLL = "ROLL"
    CLOSE_OPTION = "CLOSE_OPTION"
    CLOSE_POSITION = "CLOSE_POSITION"


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "Low IV"
    NEUTRAL = "Neutral IV"
    HIGH = "High IV"


class PositionRegime(Enum):
    """Position management regime"""
    ASSIGNMENT_CONTROL = "Assignment Control Mode"
    RECOVERY = "Recovery Mode"


@dataclass
class PositionInputs:
    """Position and market state inputs"""
    ticker: str
    shares_held: float
    stock_cost_basis: float
    option_strike: float
    option_expiration: datetime
    option_entry_premium: float
    current_option_mark: float
    current_stock_price: float
    theta: Optional[float] = None
    delta: Optional[float] = None
    earnings_date: Optional[datetime] = None
    dividend_date: Optional[datetime] = None
    iv: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    historical_volatility_30d: Optional[float] = None
    gamma: Optional[float] = None
    risk_free_rate: float = 0.05  # Default 5%


@dataclass
class DerivedMetrics:
    """Calculated metrics from position inputs"""
    dte: float
    stock_pl: float
    option_pl: float
    total_pl: float
    premium_captured_pct: float
    extrinsic_remaining: float
    break_even_price: float
    intrinsic_value: float
    gamma_risk_proxy: float
    annualized_yield: float
    return_on_capital: float
    distance_from_strike_pct: float
    distance_from_cost_basis_pct: float
    delta: float
    theta: float
    iv_hv_spread: Optional[float] = None
    expected_move: Optional[float] = None


@dataclass
class VolatilityState:
    """Volatility regime classification"""
    regime: VolatilityRegime
    iv_rank: float
    iv_percentile: float
    iv_hv_spread: float
    expected_move: float


@dataclass
class PositionState:
    """Position regime classification"""
    regime: PositionRegime
    is_below_cost_basis_threshold: bool


@dataclass
class RollCandidate:
    """Potential roll candidate contract"""
    strike: float
    expiration: datetime
    dte: float
    premium: float
    delta: float
    annualized_yield: float
    yield_per_unit_risk: float
    cost_basis_reduction: float
    distance_to_cost_basis_pct: float
    net_credit: float
    is_valid: bool
    validation_errors: List[str]


# ============================================================================
# MODULE 1: BLACK-SCHOLES OPTION PRICING
# ============================================================================

class BlackScholes:
    """Black-Scholes option pricing model"""
    
    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call') -> float:
        """
        Calculate option delta using Black-Scholes
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            Delta value
        """
        if T <= 0:
            return 1.0 if S > K else 0.0 if option_type == 'call' else -1.0 if S < K else 0.0
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    @staticmethod
    def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_theta(S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call') -> float:
        """Calculate option theta (annualized)"""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        return theta / 365  # Daily theta
    
    @staticmethod
    def calculate_premium(S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str = 'call') -> float:
        """Calculate option premium"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================================
# MODULE 2: DERIVED METRICS ENGINE
# ============================================================================

class DerivedMetricsEngine:
    """Calculates all derived metrics from position inputs"""
    
    def __init__(self, bs_model: BlackScholes):
        self.bs = bs_model
    
    def calculate(self, inputs: PositionInputs) -> DerivedMetrics:
        """Calculate all derived metrics"""
        
        # Days to expiration
        dte = max(0, (inputs.option_expiration - datetime.now()).days)
        
        # P&L calculations
        stock_pl = (inputs.current_stock_price - inputs.stock_cost_basis) * inputs.shares_held
        # Covered call is a short call: profit increases as mark decreases.
        option_pl = (inputs.option_entry_premium - inputs.current_option_mark) * 100  # per contract
        total_pl = stock_pl + option_pl
        
        # Premium capture
        premium_captured_pct = 1 - (inputs.current_option_mark / inputs.option_entry_premium) if inputs.option_entry_premium > 0 else 0
        
        # Option metrics
        intrinsic_value = max(0, inputs.current_stock_price - inputs.option_strike)
        extrinsic_remaining = inputs.current_option_mark - intrinsic_value
        
        # Break-even
        break_even_price = inputs.stock_cost_basis - (inputs.option_entry_premium / 100)  # per share premium collected
        
        # Gamma risk proxy (if gamma not provided)
        if inputs.gamma is not None:
            gamma_risk_proxy = inputs.gamma * 100  # Scale for 1-point move
        else:
            # Approximate delta change for $1 move
            if inputs.delta is not None:
                # Simple approximation: delta changes by ~0.1 per 10% moneyness change
                gamma_risk_proxy = abs(inputs.delta * 0.1)
            else:
                gamma_risk_proxy = 0.05  # Default conservative estimate
        
        # Annualized yield
        annualized_yield = (inputs.current_option_mark / inputs.current_stock_price) * (365 / dte) if dte > 0 else 0
        
        # Return on capital
        capital_employed = inputs.stock_cost_basis * inputs.shares_held
        roc = total_pl / capital_employed if capital_employed > 0 else 0
        
        # Distance metrics
        distance_from_strike_pct = (inputs.option_strike - inputs.current_stock_price) / inputs.current_stock_price
        distance_from_cost_basis_pct = (inputs.current_stock_price - inputs.stock_cost_basis) / inputs.stock_cost_basis
        
        # Delta and theta (calculate if not provided)
        delta = inputs.delta
        theta = inputs.theta
        
        if delta is None and inputs.iv is not None:
            T = dte / 365
            delta = self.bs.calculate_delta(
                inputs.current_stock_price, inputs.option_strike, T, 
                inputs.risk_free_rate, inputs.iv, 'call'
            )
        
        if theta is None and inputs.iv is not None:
            T = dte / 365
            theta = self.bs.calculate_theta(
                inputs.current_stock_price, inputs.option_strike, T,
                inputs.risk_free_rate, inputs.iv, 'call'
            )
        
        # IV-HV spread
        iv_hv_spread = None
        if inputs.iv is not None and inputs.historical_volatility_30d is not None:
            iv_hv_spread = inputs.iv - inputs.historical_volatility_30d
        
        # Expected move
        expected_move = None
        if inputs.iv is not None:
            expected_move = inputs.current_stock_price * inputs.iv * np.sqrt(dte / 365)
        
        return DerivedMetrics(
            dte=dte,
            stock_pl=stock_pl,
            option_pl=option_pl,
            total_pl=total_pl,
            premium_captured_pct=premium_captured_pct,
            extrinsic_remaining=extrinsic_remaining,
            break_even_price=break_even_price,
            intrinsic_value=intrinsic_value,
            gamma_risk_proxy=gamma_risk_proxy,
            annualized_yield=annualized_yield,
            return_on_capital=roc,
            distance_from_strike_pct=distance_from_strike_pct,
            distance_from_cost_basis_pct=distance_from_cost_basis_pct,
            delta=delta if delta is not None else 0.0,
            theta=theta if theta is not None else 0.0,
            iv_hv_spread=iv_hv_spread,
            expected_move=expected_move
        )


# ============================================================================
# MODULE 3: VOLATILITY & REGIME CLASSIFICATION
# ============================================================================

class VolatilityRegimeClassifier:
    """Classifies volatility and position regimes"""
    
    def classify_volatility(self, inputs: PositionInputs) -> VolatilityState:
        """Classify volatility regime"""
        
        # Default values if not provided
        iv_rank = inputs.iv_rank if inputs.iv_rank is not None else 50
        iv_percentile = inputs.iv_percentile if inputs.iv_percentile is not None else 50
        
        # Determine regime
        if iv_rank < 20:
            regime = VolatilityRegime.LOW
        elif iv_rank <= 50:
            regime = VolatilityRegime.NEUTRAL
        else:
            regime = VolatilityRegime.HIGH
        
        # Calculate IV-HV spread
        iv_hv_spread = 0
        if inputs.iv is not None and inputs.historical_volatility_30d is not None:
            iv_hv_spread = inputs.iv - inputs.historical_volatility_30d
        
        # Calculate expected move
        expected_move = 0
        if inputs.iv is not None and inputs.current_stock_price is not None:
            dte = max(1, (inputs.option_expiration - datetime.now()).days)
            expected_move = inputs.current_stock_price * inputs.iv * np.sqrt(dte / 365)
        
        return VolatilityState(
            regime=regime,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_hv_spread=iv_hv_spread,
            expected_move=expected_move
        )
    
    def classify_position(self, inputs: PositionInputs, metrics: DerivedMetrics) -> PositionState:
        """Classify position regime"""
        
        # Check if below cost basis threshold (90% of cost basis)
        is_below_threshold = inputs.current_stock_price < (0.9 * inputs.stock_cost_basis)
        
        # Determine regime
        if is_below_threshold:
            regime = PositionRegime.RECOVERY
        else:
            regime = PositionRegime.ASSIGNMENT_CONTROL
        
        return PositionState(
            regime=regime,
            is_below_cost_basis_threshold=is_below_threshold
        )


# ============================================================================
# MODULE 4: DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """Generates trading signals based on position metrics"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def generate_signal(self, inputs: PositionInputs, metrics: DerivedMetrics, 
                       vol_state: VolatilityState, pos_state: PositionState) -> Signal:
        """
        Generate trading signal based on all inputs
        
        Returns:
            Signal enum value
        """
        
        # CLOSE_POSITION - Rare, only in extreme cases
        if self._should_close_position(inputs, metrics):
            return Signal.CLOSE_POSITION
        
        # CLOSE_OPTION
        if self._should_close_option(inputs, metrics, vol_state):
            return Signal.CLOSE_OPTION
        
        # ROLL
        if self._should_roll(inputs, metrics, vol_state, pos_state):
            return Signal.ROLL
        
        # Default to KEEP
        return Signal.KEEP
    
    def _should_close_position(self, inputs: PositionInputs, metrics: DerivedMetrics) -> bool:
        """Check if position should be closed entirely"""
        
        # Total drawdown exceeds 20% (portfolio tolerance threshold)
        total_drawdown = -metrics.total_pl / (inputs.stock_cost_basis * inputs.shares_held)
        if total_drawdown > 0.20:
            return True
        
        # Income fails to offset depreciation over 90 days
        if metrics.dte < 90 and metrics.premium_captured_pct < 0.3 and metrics.distance_from_cost_basis_pct < -0.15:
            return True
        
        return False
    
    def _should_close_option(self, inputs: PositionInputs, metrics: DerivedMetrics, 
                            vol_state: VolatilityState) -> bool:
        """Check if option should be closed (buy back)"""
        
        # Extrinsic value negligible (< $0.10)
        if metrics.extrinsic_remaining < 0.10:
            return True
        
        # Delta < 0.05 (deep OTM)
        if metrics.delta < 0.05:
            return True
        
        # Earnings within 5-7 days
        if inputs.earnings_date:
            days_to_earnings = (inputs.earnings_date - datetime.now()).days
            if 5 <= days_to_earnings <= 7:
                return True
        
        # Gamma risk high near expiration (DTE < 3 and near ATM)
        if metrics.dte < 3 and abs(metrics.distance_from_strike_pct) < 0.02:
            return True
        
        return False
    
    def _should_roll(self, inputs: PositionInputs, metrics: DerivedMetrics,
                    vol_state: VolatilityState, pos_state: PositionState) -> bool:
        """Check if option should be rolled"""
        
        # ===== Assignment Risk Triggers (Assignment Control Mode only) =====
        if pos_state.regime == PositionRegime.ASSIGNMENT_CONTROL:
            # Delta > 0.45
            if metrics.delta > 0.45:
                return True
            
            # Extrinsic value < 25% of original premium
            if metrics.extrinsic_remaining < (0.25 * inputs.option_entry_premium):
                return True
            
            # DTE < 21
            if metrics.dte < 21:
                return True
            
            # Stock near/above strike (within 2%)
            if metrics.distance_from_strike_pct <= 0.02:
                return True
        
        # ===== Yield Efficiency Upgrade (Both Modes) =====
        # Premium captured high (>70%) and remaining yield low
        if metrics.premium_captured_pct > 0.7:
            remaining_daily_yield = metrics.annualized_yield / 365
            if remaining_daily_yield < self.risk_free_rate / 365:
                return True
        
        # ===== Volatility Expansion Opportunity =====
        # IV expanded significantly (>20% increase)
        if vol_state.iv_hv_spread > 0.2:
            return True
        
        # New contract offers superior yield (will be checked in optimizer)
        
        return False


# ============================================================================
# MODULE 5: ROLLING OPTIMIZER
# ============================================================================

class RollingOptimizer:
    """Optimizes roll candidates with production-safe constraints"""
    
    def __init__(self, bs_model: BlackScholes, risk_free_rate: float = 0.05):
        self.bs = bs_model
        self.risk_free_rate = risk_free_rate
    
    def find_optimal_roll(self, inputs: PositionInputs, metrics: DerivedMetrics,
                         vol_state: VolatilityState, pos_state: PositionState,
                         available_strikes: List[float], available_expirations: List[datetime]) -> Optional[RollCandidate]:
        """
        Find optimal roll candidate subject to all constraints
        
        Args:
            inputs: Original position inputs
            metrics: Current derived metrics
            vol_state: Volatility state
            pos_state: Position regime
            available_strikes: List of available strike prices
            available_expirations: List of available expiration dates
        
        Returns:
            Optimal RollCandidate or None if no valid candidate found
        """
        
        candidates = []
        
        for expiration in available_expirations:
            dte = (expiration - datetime.now()).days
            
            # Target DTE: 30-45 days
            if dte < 30 or dte > 45:
                continue
            
            # Earnings avoidance (if earnings date known)
            if inputs.earnings_date:
                days_before_earnings = (inputs.earnings_date - expiration).days
                if -7 <= days_before_earnings <= 7:  # Expiration within 7 days of earnings
                    continue
            
            for strike in available_strikes:
                candidate = self._evaluate_candidate(
                    inputs, metrics, vol_state, pos_state,
                    strike, expiration, dte
                )
                
                if candidate and candidate.is_valid:
                    candidates.append(candidate)
        
        if not candidates:
            return None
        
        # Select candidate maximizing yield per unit assignment risk
        optimal = max(candidates, key=lambda c: c.yield_per_unit_risk)
        
        return optimal
    
    def _evaluate_candidate(self, inputs: PositionInputs, metrics: DerivedMetrics,
                           vol_state: VolatilityState, pos_state: PositionState,
                           strike: float, expiration: datetime, dte: float) -> Optional[RollCandidate]:
        """Evaluate a single roll candidate against all constraints"""
        
        validation_errors = []
        
        # ===== MONEYNESS ANCHOR (MANDATORY) =====
        # All strikes must be anchored to CURRENT stock price
        
        # Allowed OTM distance based on regime
        if vol_state.regime == VolatilityRegime.HIGH:
            max_otm_pct = 0.25  # 25% in high IV
        else:
            max_otm_pct = 0.15  # 15% in normal conditions
        
        # Calculate OTM percentage (positive means strike above spot)
        otm_pct = (strike - inputs.current_stock_price) / inputs.current_stock_price
        
        # NEVER exceed 30% OTM
        if otm_pct > 0.30:
            validation_errors.append(f"Strike {strike} exceeds 30% OTM limit ({otm_pct:.1%})")
        
        # Check regime-specific limit
        if otm_pct > max_otm_pct:
            validation_errors.append(f"Strike {strike} exceeds regime OTM limit of {max_otm_pct:.0%}")
        
        # ===== MODE-SPECIFIC DELTA TARGETS =====
        # Calculate delta for this candidate
        T = dte / 365
        if inputs.iv is not None:
            delta = self.bs.calculate_delta(
                inputs.current_stock_price, strike, T,
                inputs.risk_free_rate, inputs.iv, 'call'
            )
        else:
            # Default IV if not provided (use 0.3 as conservative estimate)
            delta = self.bs.calculate_delta(
                inputs.current_stock_price, strike, T,
                inputs.risk_free_rate, 0.3, 'call'
            )
        
        # Set delta targets based on regime
        if pos_state.regime == PositionRegime.ASSIGNMENT_CONTROL:
            max_delta = 0.35
            target_delta_min = 0.18
            target_delta_max = 0.30
        else:  # Recovery Mode
            max_delta = 0.30
            target_delta_min = 0.15
            target_delta_max = 0.25
        
        if delta > max_delta:
            validation_errors.append(f"Delta {delta:.3f} exceeds maximum {max_delta}")
        
        # ===== NET CREDIT REQUIRED =====
        # Calculate premium for new option
        if inputs.iv is not None:
            new_premium = self.bs.calculate_premium(
                inputs.current_stock_price, strike, T,
                inputs.risk_free_rate, inputs.iv, 'call'
            )
        else:
            new_premium = self.bs.calculate_premium(
                inputs.current_stock_price, strike, T,
                inputs.risk_free_rate, 0.3, 'call'
            )
        
        # Net credit: sell new (get premium) - buy back old (pay current mark)
        net_credit = new_premium - inputs.current_option_mark
        
        if net_credit <= 0:
            validation_errors.append(f"Net credit {net_credit:.3f} is not positive")
        
        # ===== COST BASIS REDUCTION IN RECOVERY MODE =====
        if pos_state.regime == PositionRegime.RECOVERY:
            # Prohibit aggressive roll-up above cost basis
            if strike > inputs.stock_cost_basis:
                # Check if premium efficiency justifies (at least 15% annualized yield)
                annualized_yield = (new_premium / inputs.current_stock_price) * (365 / dte)
                if annualized_yield < 0.15:
                    validation_errors.append(f"Recovery mode: strike above cost basis with insufficient yield {annualized_yield:.1%}")
        
        # ===== CANDIDATE METRICS =====
        annualized_yield = (new_premium / inputs.current_stock_price) * (365 / dte)
        
        # Yield per unit assignment risk (higher is better)
        yield_per_unit_risk = annualized_yield / delta if delta > 0 else 0
        
        # Cost basis reduction effect
        cost_basis_reduction = new_premium / 100  # per share reduction
        
        # Distance to cost basis
        distance_to_cost_basis_pct = (inputs.current_stock_price - inputs.stock_cost_basis) / inputs.stock_cost_basis
        
        return RollCandidate(
            strike=strike,
            expiration=expiration,
            dte=dte,
            premium=new_premium,
            delta=delta,
            annualized_yield=annualized_yield,
            yield_per_unit_risk=yield_per_unit_risk,
            cost_basis_reduction=cost_basis_reduction,
            distance_to_cost_basis_pct=distance_to_cost_basis_pct,
            net_credit=net_credit,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors
        )


# ============================================================================
# MODULE 6: CAPITAL EFFICIENCY & RISK DIAGNOSTICS
# ============================================================================

@dataclass
class RiskDiagnostics:
    """Risk and capital efficiency metrics"""
    roc: float
    yield_vs_rfr: float
    yield_vs_hv: Optional[float]
    income_vs_depreciation: float
    downside_cushion: float
    upside_cap_impact: float
    sharpe_ratio: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'ROC': f"{self.roc:.2%}",
            'Yield vs RFR': f"{self.yield_vs_rfr:.2f}x",
            'Yield vs HV': f"{self.yield_vs_hv:.2f}x" if self.yield_vs_hv else "N/A",
            'Income/Depreciation': f"{self.income_vs_depreciation:.2f}x",
            'Downside Cushion': f"{self.downside_cushion:.2%}",
            'Upside Cap Impact': f"{self.upside_cap_impact:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio else "N/A"
        }


class RiskDiagnosticsEngine:
    """Calculates capital efficiency and risk metrics"""
    
    def calculate(self, inputs: PositionInputs, metrics: DerivedMetrics,
                 vol_state: VolatilityState, pos_state: PositionState) -> RiskDiagnostics:
        """Calculate all risk diagnostics"""
        
        # ROC (already in metrics)
        roc = metrics.return_on_capital
        
        # Yield vs risk-free rate
        yield_vs_rfr = metrics.annualized_yield / inputs.risk_free_rate if inputs.risk_free_rate > 0 else float('inf')
        
        # Yield vs historical volatility
        yield_vs_hv = None
        if inputs.historical_volatility_30d is not None:
            yield_vs_hv = metrics.annualized_yield / inputs.historical_volatility_30d
        
        # Income vs capital depreciation
        total_premium_collected = inputs.option_entry_premium * 100  # per contract
        stock_depreciation = max(0, (inputs.stock_cost_basis - inputs.current_stock_price) * inputs.shares_held)
        income_vs_depreciation = total_premium_collected / stock_depreciation if stock_depreciation > 0 else float('inf')
        
        # Downside cushion (premium collected as % of stock price)
        downside_cushion = inputs.option_entry_premium / inputs.current_stock_price
        
        # Upside cap impact (strike vs current price)
        upside_cap_impact = (inputs.option_strike - inputs.current_stock_price) / inputs.current_stock_price
        
        # Sharpe ratio (simplified)
        sharpe_ratio = None
        if inputs.historical_volatility_30d is not None and inputs.risk_free_rate is not None:
            excess_return = metrics.annualized_yield - inputs.risk_free_rate
            sharpe_ratio = excess_return / inputs.historical_volatility_30d if inputs.historical_volatility_30d > 0 else 0
        
        return RiskDiagnostics(
            roc=roc,
            yield_vs_rfr=yield_vs_rfr,
            yield_vs_hv=yield_vs_hv,
            income_vs_depreciation=income_vs_depreciation,
            downside_cushion=downside_cushion,
            upside_cap_impact=upside_cap_impact,
            sharpe_ratio=sharpe_ratio
        )


# ============================================================================
# MAIN ENGINE: COVERED CALL MANAGER
# ============================================================================

class CoveredCallEngine:
    """
    Main covered call management engine
    
    Integrates all six modules to analyze positions and generate signals
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.bs = BlackScholes()
        self.metrics_engine = DerivedMetricsEngine(self.bs)
        self.vol_classifier = VolatilityRegimeClassifier()
        self.decision_engine = DecisionEngine(risk_free_rate)
        self.roll_optimizer = RollingOptimizer(self.bs, risk_free_rate)
        self.risk_engine = RiskDiagnosticsEngine()
        
        self.risk_free_rate = risk_free_rate
    
    def analyze_position(self, inputs: PositionInputs, 
                        available_strikes: Optional[List[float]] = None,
                        available_expirations: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        Complete analysis of covered call position
        
        Args:
            inputs: Position inputs
            available_strikes: List of available strikes for rolling
            available_expirations: List of available expirations for rolling
        
        Returns:
            Dictionary with analysis results formatted for output
        """
        
        # Step 1: Calculate derived metrics
        metrics = self.metrics_engine.calculate(inputs)
        
        # Step 2: Classify regimes
        vol_state = self.vol_classifier.classify_volatility(inputs)
        pos_state = self.vol_classifier.classify_position(inputs, metrics)
        
        # Step 3: Generate signal
        signal = self.decision_engine.generate_signal(inputs, metrics, vol_state, pos_state)
        
        # Step 4: Find optimal roll if signal is ROLL
        roll_candidate = None
        if signal == Signal.ROLL and available_strikes and available_expirations:
            roll_candidate = self.roll_optimizer.find_optimal_roll(
                inputs, metrics, vol_state, pos_state,
                available_strikes, available_expirations
            )
            
            # If no valid candidate found, fall back to KEEP
            if not roll_candidate:
                signal = Signal.KEEP
        
        # Step 5: Calculate risk diagnostics
        risk_diagnostics = self.risk_engine.calculate(inputs, metrics, vol_state, pos_state)
        
        # Step 6: Format output
        return self._format_output(inputs, metrics, vol_state, pos_state, 
                                   signal, roll_candidate, risk_diagnostics)
    
    def _format_output(self, inputs: PositionInputs, metrics: DerivedMetrics,
                      vol_state: VolatilityState, pos_state: PositionState,
                      signal: Signal, roll_candidate: Optional[RollCandidate],
                      risk_diagnostics: RiskDiagnostics) -> Dict[str, Any]:
        """Format analysis results for output"""
        
        # SECTION 1: SUMMARY TABLE
        summary = {
            'Recommended Action': signal.value
        }
        
        # Add roll details if applicable
        if signal == Signal.ROLL and roll_candidate:
            summary.update({
                'If ROLL: Suggested Strike': f"${roll_candidate.strike:.2f}",
                'If ROLL: Suggested Expiration': f"{roll_candidate.dte:.0f} DTE",
                'If ROLL: Net Credit': f"${roll_candidate.net_credit:.2f}",
                'If ROLL: New Delta': f"{roll_candidate.delta:.3f}",
                'If ROLL: New Annualized Yield': f"{roll_candidate.annualized_yield:.1%}"
            })

        summary.update({
            'Stock Price': f"${inputs.current_stock_price:.2f}",
            'Cost Basis': f"${inputs.stock_cost_basis:.2f}",
            'Strike': f"${inputs.option_strike:.2f}",
            'DTE': f"{metrics.dte:.0f}",
            'Stock P/L': f"${metrics.stock_pl:.2f}",
            'Option P/L': f"${metrics.option_pl:.2f}",
            'Total Position P/L': f"${metrics.total_pl:.2f}",
            '% Premium Captured': f"{metrics.premium_captured_pct:.1%}",
            'Extrinsic Remaining': f"${metrics.extrinsic_remaining:.2f}",
            'Delta': f"{metrics.delta:.3f}",
            'Assignment Risk Level': self._get_assignment_risk_level(metrics.delta),
            'IV Regime': vol_state.regime.value,
            'Position Regime': pos_state.regime.value,
            'Annualized Yield': f"{metrics.annualized_yield:.1%}"
        })
        
        # SECTION 2: REASONING (will be generated by caller)
        
        return {
            'summary': summary,
            'signal': signal,
            'roll_candidate': roll_candidate,
            'metrics': metrics,
            'vol_state': vol_state,
            'pos_state': pos_state,
            'risk_diagnostics': risk_diagnostics,
            'inputs': inputs
        }
    
    def _get_assignment_risk_level(self, delta: float) -> str:
        """Convert delta to assignment risk level"""
        if delta < 0.2:
            return "Very Low"
        elif delta < 0.3:
            return "Low"
        elif delta < 0.4:
            return "Moderate"
        elif delta < 0.5:
            return "Elevated"
        else:
            return "High"


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates human-readable reports from analysis results"""
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate complete report with table and reasoning"""
        
        output = []
        
        # SECTION 1: SUMMARY TABLE
        output.append("🔹 SECTION 1 — SUMMARY (TABLE ONLY)")
        output.append("")
        output.append("| Metric | Value |")
        output.append("| --- | --- |")
        
        for key, value in analysis['summary'].items():
            output.append(f"| {key} | {value} |")
        
        output.append("")
        output.append("🔹 SECTION 2 — REASONING (FREE TEXT REPORT)")
        output.append("")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(analysis)
        output.append(reasoning)
        
        return "\n".join(output)
    
    def _generate_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed reasoning narrative"""
        
        inputs = analysis['inputs']
        metrics = analysis['metrics']
        vol_state = analysis['vol_state']
        pos_state = analysis['pos_state']
        signal = analysis['signal']
        risk = analysis['risk_diagnostics']
        roll_candidate = analysis.get('roll_candidate')
        
        sections = []
        
        # Current position health
        health = []
        health.append("**Current Position Health**")
        
        if metrics.total_pl > 0:
            health.append(f"The position is currently profitable with a total P/L of ${metrics.total_pl:.2f}. ")
        else:
            health.append(f"The position is underwater with a total P/L of ${metrics.total_pl:.2f}. ")
        
        health.append(f"The stock is trading at ${inputs.current_stock_price:.2f}, which is ")
        if metrics.distance_from_cost_basis_pct > 0:
            health.append(f"{metrics.distance_from_cost_basis_pct:.1%} above the cost basis of ${inputs.stock_cost_basis:.2f}. ")
        else:
            health.append(f"{abs(metrics.distance_from_cost_basis_pct):.1%} below the cost basis of ${inputs.stock_cost_basis:.2f}. ")
        
        health.append(f"You have captured {metrics.premium_captured_pct:.1%} of the original premium, with ${metrics.extrinsic_remaining:.2f} extrinsic value remaining. ")
        
        sections.append("".join(health))
        
        # Volatility regime impact
        vol_text = []
        vol_text.append("**Volatility Regime Impact**\n")
        vol_text.append(f"The current volatility regime is classified as **{vol_state.regime.value}** ")
        vol_text.append(f"(IV Rank: {vol_state.iv_rank:.0f}). ")
        
        if vol_state.regime == VolatilityRegime.LOW:
            vol_text.append("Low IV environments favor holding positions and waiting for volatility expansion. Aggressive rolling is generally not recommended.")
        elif vol_state.regime == VolatilityRegime.NEUTRAL:
            vol_text.append("Neutral IV provides a balanced environment for strategic rolling based on yield efficiency rather than volatility capture.")
        else:  # HIGH
            vol_text.append("High IV creates favorable conditions for selling premium, but requires careful strike selection to manage assignment risk.")
        
        if vol_state.iv_hv_spread > 0.1:
            vol_text.append(f" The IV-HV spread of +{vol_state.iv_hv_spread:.1%} indicates implied volatility is rich relative to recent realized volatility.")
        
        sections.append("".join(vol_text))
        
        # Assignment probability
        assign_text = []
        assign_text.append("**Assignment Probability**\n")
        assign_text.append(f"With a delta of {metrics.delta:.3f}, the probability of assignment is approximately {metrics.delta:.1%}. ")
        assign_text.append(f"The stock is currently {metrics.distance_from_strike_pct:.1%} ")
        if metrics.distance_from_strike_pct > 0:
            assign_text.append("below ")
        else:
            assign_text.append("above ")
        assign_text.append(f"the ${inputs.option_strike:.2f} strike. ")
        
        if pos_state.regime == PositionRegime.ASSIGNMENT_CONTROL:
            assign_text.append("The position is in Assignment Control Mode, prioritizing assignment avoidance.")
        else:
            assign_text.append("The position is in Recovery Mode, prioritizing cost basis reduction.")
        
        sections.append("".join(assign_text))
        
        # Signal justification
        justify = []
        justify.append(f"**Justification for {signal.value} Signal**\n")
        
        if signal == Signal.KEEP:
            justify.append("The KEEP signal is recommended because the position remains efficient with acceptable risk parameters. ")
            justify.append("Extrinsic value remains sufficient relative to theta decay, gamma risk is controlled, and no earnings are imminent. ")
            justify.append("Rolling would not provide sufficient improvement in risk-adjusted yield at this time.")
        
        elif signal == Signal.ROLL:
            justify.append("The ROLL signal is triggered based on the following factors:\n")
            
            # List specific triggers
            triggers = []
            if metrics.dte < 21:
                triggers.append(f"• DTE of {metrics.dte:.0f} is below the 21-day threshold")
            if metrics.premium_captured_pct > 0.7:
                triggers.append(f"• Premium capture of {metrics.premium_captured_pct:.1%} is high")
            if metrics.delta > 0.35 and pos_state.regime == PositionRegime.ASSIGNMENT_CONTROL:
                triggers.append(f"• Delta of {metrics.delta:.3f} exceeds assignment control threshold")
            if vol_state.regime == VolatilityRegime.HIGH and vol_state.iv_hv_spread > 0.15:
                triggers.append("• Volatility expansion opportunity exists")
            
            justify.extend(triggers)
            
            if roll_candidate:
                justify.append(f"\nThe optimal roll candidate is the ${roll_candidate.strike:.2f} strike with {roll_candidate.dte:.0f} DTE. ")
                justify.append(f"This roll generates a net credit of ${roll_candidate.net_credit:.2f}, improves annualized yield from {metrics.annualized_yield:.1%} to {roll_candidate.annualized_yield:.1%}, ")
                justify.append(f"and maintains delta at {roll_candidate.delta:.3f} which is appropriate for {pos_state.regime.value}.")
        
        elif signal == Signal.CLOSE_OPTION:
            justify.append("The CLOSE_OPTION signal is recommended because the option has minimal extrinsic value remaining. ")
            justify.append(f"With extrinsic value of only ${metrics.extrinsic_remaining:.2f} and delta of {metrics.delta:.3f}, ")
            justify.append("the option provides negligible further income potential and exposes the position to gamma risk near expiration.")
        
        else:  # CLOSE_POSITION
            justify.append("The CLOSE_POSITION signal is recommended only in rare circumstances. ")
            justify.append(f"Total drawdown of {abs(metrics.return_on_capital):.1%} exceeds portfolio tolerance thresholds, ")
            justify.append("and income generation has failed to offset capital depreciation.")
        
        sections.append("".join(justify))
        
        # Trade-offs
        tradeoffs = []
        tradeoffs.append("**Trade-offs of Alternative Actions**\n")
        
        if signal != Signal.KEEP:
            tradeoffs.append("• **KEEP**: Would maintain current income stream but misses opportunity to enhance yield or reduce risk.")
        
        if signal != Signal.ROLL:
            tradeoffs.append("• **ROLL**: Could potentially improve yield or reduce assignment risk but requires transaction costs and introduces new risk parameters.")
        
        if signal != Signal.CLOSE_OPTION and signal != Signal.CLOSE_POSITION:
            tradeoffs.append("• **CLOSE_OPTION**: Would eliminate assignment risk entirely but forgoes remaining time premium.")
        
        if signal != Signal.CLOSE_POSITION:
            tradeoffs.append("• **CLOSE_POSITION**: Would liquidate entire position, freeing capital but realizing any losses and losing upside potential.")
        
        sections.append("".join(tradeoffs))
        
        # Risk outlook
        risk_text = []
        risk_text.append("**Risk Outlook Over Next 30–45 Days**\n")
        
        # Upside risk
        risk_text.append(f"**Upside Risk**: The stock could rally above the strike, leading to assignment. ")
        if roll_candidate:
            risk_text.append(f"The new ${roll_candidate.strike:.2f} strike is {((roll_candidate.strike/inputs.current_stock_price)-1):.1%} above current price. ")
        elif signal == Signal.KEEP:
            risk_text.append(f"The current ${inputs.option_strike:.2f} strike is {metrics.distance_from_strike_pct:.1%} above current price. ")
        
        # Downside protection
        risk_text.append(f"\n**Downside Protection**: The collected premium provides a cushion of {risk.downside_cushion:.2%} against a decline. ")
        
        # Volatility outlook
        if vol_state.regime == VolatilityRegime.HIGH:
            risk_text.append("\n**Volatility Risk**: With IV in a high regime, a volatility contraction could reduce option premiums. ")
        elif vol_state.regime == VolatilityRegime.LOW:
            risk_text.append("\n**Volatility Risk**: Low IV limits premium collection but also suggests lower expected movement. ")
        
        sections.append("".join(risk_text))
        
        return "\n\n".join(sections)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the Covered Call Engine"""
    
    print("=" * 80)
    print("INSTITUTIONAL COVERED CALL MANAGEMENT & ROLLING ENGINE v2.0")
    print("=" * 80)
    print("\n")
    
    # Example position
    inputs = PositionInputs(
        ticker="AAPL",
        shares_held=100,
        stock_cost_basis=95.00,
        option_strike=110.00,
        option_expiration=datetime.now() + timedelta(days=15),
        option_entry_premium=3.50,
        current_option_mark=0.65,
        current_stock_price=102.50,
        theta=0.05,
        delta=0.12,
        earnings_date=datetime.now() + timedelta(days=30),
        dividend_date=None,
        iv=0.25,
        iv_rank=35,
        iv_percentile=40,
        historical_volatility_30d=0.22,
        gamma=0.02,
        risk_free_rate=0.05
    )
    
    # Available strikes and expirations for rolling
    available_strikes = [105, 110, 115, 120, 125, 130]
    available_expirations = [
        datetime.now() + timedelta(days=30),
        datetime.now() + timedelta(days=45),
        datetime.now() + timedelta(days=60)
    ]
    
    # Initialize engine
    engine = CoveredCallEngine(risk_free_rate=0.05)
    reporter = ReportGenerator()
    
    # Run analysis
    print("Analyzing position...")
    print("-" * 40)
    
    analysis = engine.analyze_position(
        inputs, 
        available_strikes=available_strikes,
        available_expirations=available_expirations
    )
    
    # Generate report
    report = reporter.generate_report(analysis)
    
    # Print risk diagnostics
    print("\n" + "=" * 80)
    print("ADDITIONAL RISK DIAGNOSTICS")
    print("=" * 80)
    
    risk_dict = analysis['risk_diagnostics'].to_dict()
    for key, value in risk_dict.items():
        print(f"{key:25}: {value}")
    

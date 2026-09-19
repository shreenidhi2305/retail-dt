"""
recommendations.py

Generates explainable recommendations FROM the twin's actual simulation
outcomes -- not from an LLM, not generic text. Every recommendation states:
  WHAT happened, WHY (traced to scenario parameters), the RECOMMENDED
  action, and the EXPECTED EFFECT.
"""


def _pct(x):
    return f"{x * 100:.1f}%"


def build_recommendations(baseline_agg, scenario_agg, comparison, scenario_config,
                           critical_df, health_baseline, health_scenario):
    """
    Returns a list of dicts:
      {finding, cause, recommendation, expected_effect}
    grounded entirely in baseline_agg / scenario_agg / comparison numbers.
    """
    if not baseline_agg or not scenario_agg:
        return []

    recs = []

    sl_change = comparison.get("service_level_change", 0)
    if sl_change < -0.005:
        cause_parts = []
        if scenario_config.demand_multiplier != 1.0:
            cause_parts.append(f"demand increased by {(scenario_config.demand_multiplier - 1) * 100:.0f}%")
        if scenario_config.lead_time_change_days != 0:
            sign = "+" if scenario_config.lead_time_change_days > 0 else ""
            cause_parts.append(f"supplier lead time changed by {sign}{scenario_config.lead_time_change_days} days")
        cause = " while ".join(cause_parts) if cause_parts else "scenario conditions changed"
        affected_n = int((critical_df["risk_level"].isin(["HIGH", "MEDIUM"])).sum()) if critical_df is not None and not critical_df.empty else 0

        recs.append({
            "finding": f"Service level decreased by {abs(sl_change) * 100:.1f} percentage points "
                       f"(from {_pct(baseline_agg['service_level'])} to {_pct(scenario_agg['service_level'])}).",
            "cause": f"{cause.capitalize()}, exceeding the current safety-stock buffer for "
                     f"at least {affected_n} store-item combination(s) shown in the risk ranking below.",
            "recommendation": "Increase safety stock for the affected store-item pairs and/or "
                               "trigger replenishment earlier (lower the reorder point trigger margin).",
            "expected_effect": "Reduced stockout exposure and a partial recovery of service level "
                                "toward the baseline, at the cost of higher average inventory.",
        })
    elif sl_change > 0.005:
        recs.append({
            "finding": f"Service level improved by {sl_change * 100:.1f} percentage points under "
                       f"scenario '{scenario_config.name}'.",
            "cause": "Demand and/or supply conditions in this scenario were more favorable than baseline "
                     "relative to the current inventory policy.",
            "recommendation": "Current safety stock and reorder point settings are sufficient for this "
                               "scenario; no immediate policy change needed.",
            "expected_effect": "Maintained high service level without additional inventory investment.",
        })

    stockout_change = comparison.get("stockout_change", 0)
    if stockout_change > 0:
        recs.append({
            "finding": f"Stockout events increased by {stockout_change} across the simulated network "
                       f"({baseline_agg['stockout_events']} -> {scenario_agg['stockout_events']}).",
            "cause": "Simulated demand outpaced available inventory before replenishment orders arrived, "
                     "driven by the configured scenario disruption.",
            "recommendation": "Shorten the effective replenishment cycle for high-risk items (place orders "
                               "sooner, or negotiate expedited shipping with the supplier during disruptions).",
            "expected_effect": "Fewer days spent below reorder point, reducing the number of stockout events.",
        })

    unmet_change = comparison.get("unmet_demand_change", 0)
    if unmet_change > 0 and baseline_agg["total_demand"] > 0:
        pct_of_total = unmet_change / baseline_agg["total_demand"]
        recs.append({
            "finding": f"Unmet demand increased by {unmet_change:.0f} units "
                       f"(+{pct_of_total * 100:.1f}% of baseline total demand).",
            "cause": "Inventory was depleted faster than it could be replenished under this scenario's "
                     "demand/lead-time conditions.",
            "recommendation": "Raise the target service level (higher safety-stock z-factor) for the "
                               "specific store-item pairs flagged as critical below.",
            "expected_effect": "Lower unmet demand at the cost of carrying more average inventory.",
        })

    if critical_df is not None and not critical_df.empty:
        high_risk = critical_df[critical_df["risk_level"] == "HIGH"]
        if len(high_risk):
            names = ", ".join(
                f"Store {r.store}/Item {r.item}" for r in high_risk.head(5).itertuples()
            )
            recs.append({
                "finding": f"{len(high_risk)} store-item pair(s) are at HIGH stockout risk: {names}"
                           f"{' (+more)' if len(high_risk) > 5 else ''}.",
                "cause": "These pairs combine high demand volatility with insufficient safety stock "
                         "relative to the scenario's disruption severity.",
                "recommendation": "Prioritize these pairs for safety-stock increases or supplier "
                                  "diversification before rolling the scenario's conditions out in reality.",
                "expected_effect": "Targeted risk reduction where it matters most, without inflating "
                                    "inventory network-wide.",
            })

    if health_baseline.get("score") is not None and health_scenario.get("score") is not None:
        health_delta = health_scenario["score"] - health_baseline["score"]
        if health_delta < -3:
            recs.append({
                "finding": f"Business Operational Health dropped by {abs(health_delta):.1f} points "
                           f"({health_baseline['score']:.1f} -> {health_scenario['score']:.1f}).",
                "cause": "Driven primarily by the service-level and stockout-risk components under "
                         "this scenario (see score breakdown).",
                "recommendation": "Treat this scenario as a rehearsal: implement the safety-stock and "
                                  "replenishment-timing changes above BEFORE a real demand surge or "
                                  "supply disruption occurs.",
                "expected_effect": "A smaller real-world health drop if the modeled disruption actually "
                                    "occurs, because mitigations are already in place.",
            })

    if not recs:
        recs.append({
            "finding": "No material degradation detected between baseline and this scenario.",
            "cause": "The current inventory policy appears robust to the configured demand/lead-time "
                     "changes over this horizon.",
            "recommendation": "No immediate action required; continue monitoring under live conditions.",
            "expected_effect": "Stable service level and inventory levels expected to persist.",
        })

    return recs

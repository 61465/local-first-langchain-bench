"""TCO with observed output-token drift — honest reporting.

R10 n=11 paired measurement:
- v2a: output drops -7.1% mean (-5.6% median)
- HSTC: output grows +9.3% mean (+0.5% median)

This means our old TCO (input-only reduction, output fixed) OVERSTATED HSTC
savings and UNDERSTATED v2a savings.

True TCO formula:
  savings = baseline_input_cost * input_reduction%
         - baseline_output_cost * output_drift%   # subtract output overage
"""
import statistics as st
import json

OUTPUT_DRIFT = {
    'v2a_mean_pct': -7.1,
    'v2a_median_pct': -5.6,
    'hstc_mean_pct': +9.3,
    'hstc_median_pct': +0.5,
}

INPUT_REDUCTION = {
    'v2a_median_pct': 36.2,  # R11 n=86
    'hstc_median_pct': 49.2,  # R11 n=86
}

# Per-request baseline (R3 median)
TOKENS_IN = 709
TOKENS_OUT = 218

SCENARIOS = [
    ("A gpt-4o-mini", 0.15, 0.60, 1_000_000),
    ("B Sonnet 4.6", 3.00, 15.00, 5_000_000),
    ("C gpt-4o", 2.50, 10.00, 10_000_000),
    ("D Opus 4.6", 5.00, 25.00, 10_000_000),
]


def savings(input_red_pct, output_drift_pct, inp_p, out_p, req_day):
    monthly_req = req_day * 30
    base_in_cost = TOKENS_IN * monthly_req / 1_000_000 * inp_p
    base_out_cost = TOKENS_OUT * monthly_req / 1_000_000 * out_p
    # Reduction (favorable) on input
    saved_input = base_in_cost * (input_red_pct / 100)
    # Drift (unfavorable if positive) on output
    extra_output = base_out_cost * (output_drift_pct / 100)
    net_savings_mo = saved_input - extra_output
    return net_savings_mo * 12, base_in_cost, base_out_cost, saved_input * 12, -extra_output * 12


def main():
    print("=== TRUE TCO (with observed output-token drift, R10 n=11 paired) ===\n")
    print("Formula: savings = input_reduction × input_cost - output_drift × output_cost\n")

    out_data = {"formula_note": "savings = input_reduction × input_cost − output_drift × output_cost",
                "input_reduction_source": "R11 n=86",
                "output_drift_source": "R10 n=11 paired (median)",
                "scenarios": []}

    for scen, inp_p, out_p, req_day in SCENARIOS:
        print(f"[{scen}] input=${inp_p}/M, output=${out_p}/M, {req_day:,} req/day")
        for method, in_red, out_dr in [
            ("v2a (median)", INPUT_REDUCTION['v2a_median_pct'], OUTPUT_DRIFT['v2a_median_pct']),
            ("HSTC (median)", INPUT_REDUCTION['hstc_median_pct'], OUTPUT_DRIFT['hstc_median_pct']),
            ("HSTC (mean, conservative)", INPUT_REDUCTION['hstc_median_pct'], OUTPUT_DRIFT['hstc_mean_pct']),
        ]:
            total_yr, bic, boc, sin_yr, sout_yr = savings(in_red, out_dr, inp_p, out_p, req_day)
            print(f"  {method:30s}  net ${total_yr:>13,.0f}/yr  (input_saved ${sin_yr:>10,.0f} + output_delta ${sout_yr:>+10,.0f})")

        print()

        row = {"scenario": scen, "input_price_per_m": inp_p, "output_price_per_m": out_p, "req_per_day": req_day}
        for method, in_red, out_dr, key in [
            ("v2a_median", INPUT_REDUCTION['v2a_median_pct'], OUTPUT_DRIFT['v2a_median_pct'], 'v2a'),
            ("hstc_median", INPUT_REDUCTION['hstc_median_pct'], OUTPUT_DRIFT['hstc_median_pct'], 'hstc_median'),
            ("hstc_mean_conservative", INPUT_REDUCTION['hstc_median_pct'], OUTPUT_DRIFT['hstc_mean_pct'], 'hstc_mean_conservative'),
        ]:
            total_yr, bic, boc, sin_yr, sout_yr = savings(in_red, out_dr, inp_p, out_p, req_day)
            row[key] = {"savings_yr": total_yr, "input_saved": sin_yr, "output_delta": sout_yr}
        out_data["scenarios"].append(row)

    with open("D:/project/local-first-langchain-bench/results/tco_true.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("✓ Written to results/tco_true.json")


if __name__ == "__main__":
    main()

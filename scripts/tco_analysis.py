#!/usr/bin/env python3
"""
TCO (Total Cost of Ownership) Analysis for LLM Deployment
Compares C INT8-AVX2, PyTorch CPU, and GPU deployments across multiple scenarios
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SystemConfig:
    """Configuration for a deployment system"""
    name: str
    latency_ms: float          # milliseconds per inference (seq_len=16)
    throughput_toks: float     # tokens per second
    memory_mb: float           # model weight memory in MB
    hardware_cost: float       # upfront hardware cost in USD
    cloud_cost_per_hour: float # cloud instance cost per hour
    power_watts: float         # average power consumption
    dev_hours_per_month: float # developer/ops time required


# System configurations based on benchmarks
SYSTEMS = {
    "c_int8_avx2": SystemConfig(
        name="C INT8-AVX2 (CPU)",
        latency_ms=0.275,
        throughput_toks=3636,  # 1 / (0.275 / 1000)
        memory_mb=0.50,
        hardware_cost=400,      # mid-range CPU server
        cloud_cost_per_hour=0.0416,  # AWS t3.medium
        power_watts=20,
        dev_hours_per_month=5   # minimal tuning needed
    ),
    "pytorch_cpu": SystemConfig(
        name="PyTorch CPU (FP32)",
        latency_ms=2.355,
        throughput_toks=425,   # 1 / (2.355 / 1000)
        memory_mb=2.01,
        hardware_cost=400,
        cloud_cost_per_hour=0.0416,  # AWS t3.medium
        power_watts=35,
        dev_hours_per_month=2   # framework handles most complexity
    ),
    "gpu_t4": SystemConfig(
        name="GPU (NVIDIA T4)",
        latency_ms=0.05,       # estimated
        throughput_toks=20000, # estimated
        memory_mb=2.01,
        hardware_cost=2500,
        cloud_cost_per_hour=0.526,  # AWS g4dn.xlarge
        power_watts=70,
        dev_hours_per_month=8   # GPU optimization, CUDA tuning
    ),
    "gpu_a100": SystemConfig(
        name="GPU (NVIDIA A100)",
        latency_ms=0.02,       # estimated
        throughput_toks=50000, # estimated
        memory_mb=2.01,
        hardware_cost=12000,
        cloud_cost_per_hour=3.06,  # AWS p4d.xlarge
        power_watts=300,
        dev_hours_per_month=10  # enterprise-grade tuning
    )
}


@dataclass
class DeploymentScenario:
    """Deployment scenario parameters"""
    name: str
    tokens_per_month: int
    deployment_type: str  # "edge", "cloud", "serverless"
    hours_per_day: float  # how many hours system is running
    developer_hourly_rate: float


SCENARIOS = {
    "edge": DeploymentScenario(
        name="Edge Deployment (IoT/Mobile)",
        tokens_per_month=10_000_000,      # 10M tokens/month
        deployment_type="edge",
        hours_per_day=2,                   # intermittent usage
        developer_hourly_rate=100
    ),
    "serverless": DeploymentScenario(
        name="Serverless (AWS Lambda)",
        tokens_per_month=100_000_000,     # 100M tokens/month
        deployment_type="serverless",
        hours_per_day=8,                   # variable load
        developer_hourly_rate=100
    ),
    "cloud": DeploymentScenario(
        name="Cloud Batch Processing",
        tokens_per_month=1_000_000_000,   # 1B tokens/month
        deployment_type="cloud",
        hours_per_day=24,                  # always-on
        developer_hourly_rate=100
    ),
    "enterprise": DeploymentScenario(
        name="Enterprise On-Prem",
        tokens_per_month=100_000_000_000, # 100B tokens/month
        deployment_type="cloud",
        hours_per_day=24,
        developer_hourly_rate=150
    )
}


def calculate_tco(system: SystemConfig, scenario: DeploymentScenario, 
                  months: int = 12, electricity_kwh_cost: float = 0.12) -> Dict:
    """
    Calculate Total Cost of Ownership for a system in a given scenario
    
    Returns dict with cost breakdown
    """
    
    # Calculate runtime hours per month
    hours_per_month = scenario.hours_per_day * 30
    
    # 1. Hardware cost (amortized over 3 years for on-prem, 0 for cloud)
    if scenario.deployment_type == "edge":
        hardware_monthly = system.hardware_cost / 36  # 3 year amortization
    else:
        hardware_monthly = 0  # cloud uses rental
    
    # 2. Compute cost
    if scenario.deployment_type in ["cloud", "serverless"]:
        compute_monthly = system.cloud_cost_per_hour * hours_per_month
    else:
        compute_monthly = 0  # edge uses owned hardware
    
    # 3. Power cost (electricity)
    # kWh per month = (watts / 1000) * hours_per_month
    kwh_per_month = (system.power_watts / 1000) * hours_per_month
    power_monthly = kwh_per_month * electricity_kwh_cost
    
    # 4. Cooling cost (30% of power for datacenter, 0 for edge)
    if scenario.deployment_type == "edge":
        cooling_monthly = 0
    else:
        cooling_monthly = power_monthly * 0.3
    
    # 5. Memory/storage cost (estimate $5/GB/year for cloud storage)
    memory_monthly = (system.memory_mb / 1024) * 5 / 12
    
    # 6. Developer/ops time
    dev_monthly = scenario.developer_hourly_rate * system.dev_hours_per_month
    
    # Total monthly TCO
    total_monthly = (hardware_monthly + compute_monthly + power_monthly + 
                     cooling_monthly + memory_monthly + dev_monthly)
    
    # Cost per token
    cost_per_token = total_monthly / scenario.tokens_per_month
    cost_per_million_tokens = cost_per_token * 1_000_000
    
    # Performance per dollar
    perf_per_dollar = system.throughput_toks / (total_monthly / (hours_per_month / 24))
    
    return {
        "system": system.name,
        "scenario": scenario.name,
        "hardware_monthly": hardware_monthly,
        "compute_monthly": compute_monthly,
        "power_monthly": power_monthly,
        "cooling_monthly": cooling_monthly,
        "memory_monthly": memory_monthly,
        "dev_monthly": dev_monthly,
        "total_monthly": total_monthly,
        "total_annual": total_monthly * months,
        "cost_per_token": cost_per_token,
        "cost_per_million_tokens": cost_per_million_tokens,
        "perf_per_dollar": perf_per_dollar,
        "throughput_toks": system.throughput_toks,
        "latency_ms": system.latency_ms
    }


def generate_tco_table(scenario_name: str, results_dir: str = "results/tco"):
    """Generate TCO comparison table for a scenario"""
    scenario = SCENARIOS[scenario_name]
    
    results = []
    for system_name, system in SYSTEMS.items():
        tco = calculate_tco(system, scenario)
        results.append(tco)
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON
    output_file = os.path.join(results_dir, f"tco_{scenario_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print(f"\n{'='*80}")
    print(f"TCO Analysis: {scenario.name}")
    print(f"Volume: {scenario.tokens_per_month:,} tokens/month")
    print(f"{'='*80}\n")
    
    print(f"{'System':<25} {'Latency (ms)':<15} {'Throughput':<15} {'$/M tokens':<15} {'TCO/year':<15}")
    print("-" * 85)
    
    for r in results:
        print(f"{r['system']:<25} {r['latency_ms']:<15.3f} {r['throughput_toks']:<15,.0f} "
              f"${r['cost_per_million_tokens']:<14.4f} ${r['total_annual']:<14,.0f}")
    
    print("\n")
    return results


def plot_tco_comparison(scenario_name: str, results_dir: str = "results/tco"):
    """Generate TCO comparison plots"""
    scenario = SCENARIOS[scenario_name]
    
    results = []
    for system_name, system in SYSTEMS.items():
        tco = calculate_tco(system, scenario)
        results.append(tco)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'TCO Analysis: {scenario.name}\n{scenario.tokens_per_month:,} tokens/month', 
                 fontsize=16, fontweight='bold')
    
    systems = [r['system'] for r in results]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    # 1. Total Annual TCO
    annual_costs = [r['total_annual'] for r in results]
    ax1.bar(range(len(systems)), annual_costs, color=colors)
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels(systems, rotation=45, ha='right')
    ax1.set_ylabel('Annual TCO ($)')
    ax1.set_title('Total Cost of Ownership (12 months)')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(annual_costs):
        ax1.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    # 2. Cost Breakdown (stacked bar)
    cost_components = ['hardware_monthly', 'compute_monthly', 'power_monthly', 
                       'cooling_monthly', 'dev_monthly']
    component_labels = ['Hardware', 'Compute', 'Power', 'Cooling', 'DevOps']
    
    bottom = np.zeros(len(systems))
    for comp, label in zip(cost_components, component_labels):
        values = [r[comp] * 12 for r in results]  # annualize
        ax2.bar(range(len(systems)), values, bottom=bottom, label=label)
        bottom += values
    
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels(systems, rotation=45, ha='right')
    ax2.set_ylabel('Annual Cost ($)')
    ax2.set_title('Cost Breakdown by Component')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Performance per Dollar
    perf_per_dollar = [r['perf_per_dollar'] for r in results]
    ax3.bar(range(len(systems)), perf_per_dollar, color=colors)
    ax3.set_xticks(range(len(systems)))
    ax3.set_xticklabels(systems, rotation=45, ha='right')
    ax3.set_ylabel('Tokens/sec per $1/day')
    ax3.set_title('Performance per Dollar (Higher = Better)')
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(perf_per_dollar):
        ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # 4. Cost per Million Tokens
    cost_per_million = [r['cost_per_million_tokens'] for r in results]
    ax4.bar(range(len(systems)), cost_per_million, color=colors)
    ax4.set_xticks(range(len(systems)))
    ax4.set_xticklabels(systems, rotation=45, ha='right')
    ax4.set_ylabel('Cost ($/1M tokens)')
    ax4.set_title('Cost per Million Tokens (Lower = Better)')
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(cost_per_million):
        ax4.text(i, v, f'${v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"tco_comparison_{scenario_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    plt.close()


def plot_all_scenarios_summary(results_dir: str = "results/tco"):
    """Generate summary plot comparing all scenarios"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('TCO Summary: All Scenarios', fontsize=16, fontweight='bold')
    
    scenario_names = list(SCENARIOS.keys())
    system_names = list(SYSTEMS.keys())
    
    # For each scenario, calculate TCO for each system
    data = {sys: [] for sys in system_names}
    
    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        for system_name, system in SYSTEMS.items():
            tco = calculate_tco(system, scenario)
            data[system_name].append(tco['total_annual'])
    
    # Plot 1: Grouped bar chart of annual TCO
    x = np.arange(len(scenario_names))
    width = 0.2
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    for i, (system_name, values) in enumerate(data.items()):
        offset = (i - len(system_names)/2) * width
        ax1.bar(x + offset, values, width, label=SYSTEMS[system_name].name, color=colors[i])
    
    ax1.set_xlabel('Deployment Scenario')
    ax1.set_ylabel('Annual TCO ($)')
    ax1.set_title('Annual TCO by Scenario')
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCENARIOS[s].name.split('(')[0].strip() for s in scenario_names], 
                         rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')  # log scale due to wide range
    
    # Plot 2: Cost per million tokens
    data_per_million = {sys: [] for sys in system_names}
    
    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        for system_name, system in SYSTEMS.items():
            tco = calculate_tco(system, scenario)
            data_per_million[system_name].append(tco['cost_per_million_tokens'])
    
    for i, (system_name, values) in enumerate(data_per_million.items()):
        offset = (i - len(system_names)/2) * width
        ax2.bar(x + offset, values, width, label=SYSTEMS[system_name].name, color=colors[i])
    
    ax2.set_xlabel('Deployment Scenario')
    ax2.set_ylabel('Cost ($/1M tokens)')
    ax2.set_title('Cost per Million Tokens by Scenario')
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIOS[s].name.split('(')[0].strip() for s in scenario_names],
                         rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "tco_summary_all_scenarios.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize TCO for LLM deployments')
    parser.add_argument('--scenarios', type=str, default='all',
                        help='Comma-separated list of scenarios (edge,serverless,cloud,enterprise) or "all"')
    parser.add_argument('--output-dir', type=str, default='results/tco',
                        help='Output directory for results')
    parser.add_argument('--electricity-cost', type=float, default=0.12,
                        help='Electricity cost per kWh (default: $0.12)')
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.scenarios == 'all':
        scenarios_to_run = list(SCENARIOS.keys())
    else:
        scenarios_to_run = [s.strip() for s in args.scenarios.split(',')]
    
    print("="*80)
    print("LLM Deployment TCO Analysis")
    print("="*80)
    
    # Run analysis for each scenario
    for scenario_name in scenarios_to_run:
        if scenario_name not in SCENARIOS:
            print(f"Warning: Unknown scenario '{scenario_name}', skipping")
            continue
        
        # Generate table
        results = generate_tco_table(scenario_name, args.output_dir)
        
        # Generate plots
        plot_tco_comparison(scenario_name, args.output_dir)
    
    # Generate summary plot
    if len(scenarios_to_run) > 1:
        plot_all_scenarios_summary(args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to:", args.output_dir)
    print("="*80)


if __name__ == "__main__":
    main()

"""Real (non-synthetic) smoke-test plumbing for the T24 retry.

Loads and validates Alpamayo-native candidates cached from real inference, without ever generating,
perturbing, or duplicating trajectories. No fallback candidates are ever produced here.
"""

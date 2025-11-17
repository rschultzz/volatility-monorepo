# scripts/debug_which_monies.py
import inspect
import packages.shared.ingest.monies_ingest as mi

print("monies_ingest file path:")
print(mi.__file__)
print("\n--- upsert_from_dashboard_minute source ---\n")
print(inspect.getsource(mi.upsert_from_dashboard_minute))

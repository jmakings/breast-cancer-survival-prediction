import yaml
import os

# Path to your exported environment YAML
input_yaml = os.path.expanduser("~/bc_survival_env.yml")
output_yaml = os.path.expanduser("~/bc_survival_clean.yml")

# Load YAML
with open(input_yaml, "r") as f:
    env = yaml.safe_load(f)

# 1️⃣ Change Python version to 3.11
if "dependencies" in env:
    new_deps = []
    for dep in env["dependencies"]:
        if isinstance(dep, str):
            # Remove build hash if exists
            parts = dep.split("=")
            pkg_name = parts[0]
            pkg_version = parts[1] if len(parts) > 1 else None
            if pkg_name.lower() == "python":
                pkg_version = "3.11"
            dep_clean = f"{pkg_name}={pkg_version}" if pkg_version else pkg_name
            new_deps.append(dep_clean)
        else:
            new_deps.append(dep)
    env["dependencies"] = new_deps

# 2️⃣ Remove exact build numbers
cleaned_deps = []
for dep in env["dependencies"]:
    if isinstance(dep, str):
        parts = dep.split("=")
        if len(parts) > 2:
            dep = "=".join(parts[:2])
        cleaned_deps.append(dep)
    else:
        cleaned_deps.append(dep)
env["dependencies"] = cleaned_deps

# 3️⃣ Save cleaned YAML
with open(output_yaml, "w") as f:
    yaml.dump(env, f, default_flow_style=False)

print(f"Cleaned YAML saved to {output_yaml}")

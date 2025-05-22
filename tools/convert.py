import json

def fix_widget_state_key(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    widgets = metadata.get("widgets")

    if widgets is not None:
        app_data = widgets.get("application/vnd.jupyter.widget-state+json", {})
        if "state" not in app_data:
            app_data["state"] = {}
            widgets["application/vnd.jupyter.widget-state+json"] = app_data
            metadata["widgets"] = widgets
            data["metadata"] = metadata

            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✅ Fixed: 'state' key added to widgets in {notebook_path}")
        else:
            print(f"✅ No change needed: 'state' key already present in {notebook_path}")
    else:
        print(f"ℹ️ No widgets metadata found in {notebook_path}")

# 사용 예시
fix_widget_state_key("basic_week8_invalid.ipynb")

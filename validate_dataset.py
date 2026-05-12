import json
import re

# Define your allowed schema from the screenshot
ALLOWED_FUNCTIONS = ["set_mode", "start_cycle", "stop_cycle", "get_telemetry", "set_spin_speed", "get_status"]

def validate_dataset(file_path):
    errors = 0
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                text = data['text']
                
                # Check for ChatML structure
                if "<|action|>" not in text:
                    print(f"Row {line_num}: Missing <|im_start|>action")
                    errors += 1
                    continue

                # Extract the function call name
                action_match = re.search(r"<\|action\|>\n(.*?)\(", text)
                if action_match:
                    func_name = action_match.group(1).strip()
                    if func_name not in ALLOWED_FUNCTIONS:
                        print(f"Row {line_num}: Invalid function name '{func_name}'")
                        errors += 1
                else:
                    print(f"Row {line_num}: Could not parse function call")
                    errors += 1
                    
            except Exception as e:
                print(f"Row {line_num}: Invalid JSON format - {e}")
                errors += 1

    print(f"\nValidation complete. Found {errors} errors.")

if __name__ == "__main__":
    # Point this to your specific file name
    validate_dataset("train_data_final.jsonl")
# Run this after you save your data to 'laundry_data.jsonl'
# validate_dataset('laundry_data.jsonl')
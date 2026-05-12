import json

def unify_data_format(input_file, output_file):
    fixed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            text = data['text']
            
            # Check if this is the old format
            if "<|im_start|>action" in text:
                # 1. Replace the end/start markers between thought and action
                text = text.replace("<|im_end|>\n<|im_start|>action", "\n<|action|>")
                
                # 2. Replace the end/start markers between action and answer
                text = text.replace("<|im_end|>\n<|im_start|>answer", "\n<|answer|>")
                
                data['text'] = text
                fixed_count += 1
            
            f_out.write(json.dumps(data) + '\n')

    print(f"Successfully unified {fixed_count} rows.")
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    unify_data_format("train_data.jsonl", "train_data_final.jsonl")
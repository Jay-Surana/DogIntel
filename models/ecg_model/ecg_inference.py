import ast  # to safely parse the string into Python list/dict

def extract_mean_value(column):
    def safe_mean(row):
        try:
            items = ast.literal_eval(row)  # safely convert string to list of dicts
            values = [d['value'] for d in items if isinstance(d, dict) and 'value' in d]
            return sum(values) / len(values) if values else 0
        except:
            return 0  # if anything goes wrong, return 0
    return column.apply(safe_mean)

# Apply to the two problematic columns
df['segments_br_clean'] = extract_mean_value(df['segments_br'])
df['segments_hr_clean'] = extract_mean_value(df['segments_hr'])

import json

# File path
file_path = '/nfs/MoELoRA/GoLLIE/data/casie/data.jsonl'

# Read the JSONL file
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Split the data
validation_data = data[:200]
test_data = data[-2000:]

# Save the split data into separate files
with open('/nfs/MoELoRA/GoLLIE/data/casie/data.dev.jsonl', 'w') as file:
    for item in validation_data:
        file.write(json.dumps(item) + '\n')

with open('/nfs/MoELoRA/GoLLIE/data/casie/data.test.jsonl', 'w') as file:
    for item in test_data:
        file.write(json.dumps(item) + '\n')
